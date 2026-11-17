// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redis-backed subtask catalog for the RSI router.
//!
//! One `GET` returns the whole catalog as a JSON array. The client speaks only the
//! RESP commands it needs: `AUTH`, `SELECT`, and `GET`. Plain TCP and TLS URLs are
//! supported.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use libsy::{LibsyError, Result, SubtaskCatalog, SubtaskEntry};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

const MAX_LINE_BYTES: usize = 64 * 1024;
const MAX_BULK_BYTES: usize = 8 * 1024 * 1024;

/// A Redis lookup failure.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RedisCatalogError {
    #[error("redis operation timed out")]
    Timeout,
    #[error("redis i/o failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("redis returned an error: {0}")]
    Server(String),
    #[error("redis protocol error: {0}")]
    Protocol(String),
    #[error("redis URL is invalid: {0}")]
    Url(String),
    #[error("redis TLS setup failed: {0}")]
    Tls(String),
    #[error("subtask dictionary JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
}

/// Reads the whole subtask catalog with `GET <catalog_key>`.
pub(crate) struct RedisSubtaskCatalog {
    url: RedisUrl,
    catalog_key: String,
    timeout: Duration,
    tls: Option<Arc<tokio_rustls::TlsConnector>>,
}

impl RedisSubtaskCatalog {
    /// Builds a catalog client. No connection is opened until the first read.
    pub(crate) fn new(
        url: &str,
        catalog_key: impl Into<String>,
        timeout: Duration,
    ) -> std::result::Result<Self, RedisCatalogError> {
        let url = RedisUrl::parse(url)?;
        if timeout.is_zero() {
            return Err(RedisCatalogError::Url(
                "timeout must be greater than zero".to_string(),
            ));
        }
        let tls = url.tls.then(tls_connector).transpose()?;
        Ok(Self {
            url,
            catalog_key: catalog_key.into(),
            timeout,
            tls,
        })
    }

    async fn fetch_entries(&self) -> std::result::Result<Vec<SubtaskEntry>, RedisCatalogError> {
        let mut stream = self.connect().await?;
        let reply = command(&mut stream, &["GET", &self.catalog_key]).await?;
        match reply {
            Reply::Bulk(None) => Ok(Vec::new()),
            Reply::Bulk(Some(bytes)) => Ok(serde_json::from_slice(&bytes)?),
            other => Err(RedisCatalogError::Protocol(format!(
                "GET returned {}",
                other.kind()
            ))),
        }
    }

    async fn connect(&self) -> std::result::Result<Box<dyn AsyncStream>, RedisCatalogError> {
        let tcp = TcpStream::connect((self.url.host.as_str(), self.url.port)).await?;
        tcp.set_nodelay(true)?;
        let mut stream: Box<dyn AsyncStream> = match &self.tls {
            Some(connector) => {
                let server_name =
                    tokio_rustls::rustls::pki_types::ServerName::try_from(self.url.host.clone())
                        .map_err(|error| RedisCatalogError::Tls(error.to_string()))?;
                let tls = connector
                    .connect(server_name, tcp)
                    .await
                    .map_err(|error| RedisCatalogError::Tls(error.to_string()))?;
                Box::new(tls)
            }
            None => Box::new(tcp),
        };
        if let Some(password) = &self.url.password {
            let reply = match &self.url.username {
                Some(username) => command(&mut stream, &["AUTH", username, password]).await?,
                None => command(&mut stream, &["AUTH", password]).await?,
            };
            expect_ok(&reply, "AUTH")?;
        }
        if let Some(database) = self.url.database {
            let database = database.to_string();
            let reply = command(&mut stream, &["SELECT", &database]).await?;
            expect_ok(&reply, "SELECT")?;
        }
        Ok(stream)
    }
}

#[async_trait]
impl SubtaskCatalog for RedisSubtaskCatalog {
    async fn entries(&self) -> Result<Vec<SubtaskEntry>> {
        match tokio::time::timeout(self.timeout, self.fetch_entries()).await {
            Ok(Ok(entries)) => Ok(entries),
            Ok(Err(error)) => Err(LibsyError::external("subtask catalog read", error)),
            Err(_) => Err(LibsyError::external(
                "subtask catalog read",
                RedisCatalogError::Timeout,
            )),
        }
    }
}

/// A catalog for a route whose Redis settings are unavailable.
///
/// The route builds and serves requests. Every lookup fails, so the router
/// uses the flagship model until Redis is configured.
pub(crate) struct UnavailableSubtaskCatalog {
    reason: String,
}

impl UnavailableSubtaskCatalog {
    pub(crate) fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl SubtaskCatalog for UnavailableSubtaskCatalog {
    async fn entries(&self) -> Result<Vec<SubtaskEntry>> {
        Err(LibsyError::AlgorithmError {
            message: self.reason.clone(),
        })
    }
}

fn tls_connector() -> std::result::Result<Arc<tokio_rustls::TlsConnector>, RedisCatalogError> {
    let native = rustls_native_certs::load_native_certs();
    let mut roots = tokio_rustls::rustls::RootCertStore::empty();
    for certificate in native.certs {
        roots
            .add(certificate)
            .map_err(|error| RedisCatalogError::Tls(error.to_string()))?;
    }
    if roots.is_empty() {
        return Err(RedisCatalogError::Tls(
            "no native root certificates were loaded".to_string(),
        ));
    }
    let provider = tokio_rustls::rustls::crypto::aws_lc_rs::default_provider();
    let config = tokio_rustls::rustls::ClientConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions()
        .map_err(|error| RedisCatalogError::Tls(error.to_string()))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(tokio_rustls::TlsConnector::from(Arc::new(config))))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RedisUrl {
    tls: bool,
    host: String,
    port: u16,
    username: Option<String>,
    password: Option<String>,
    database: Option<u32>,
}

impl RedisUrl {
    fn parse(value: &str) -> std::result::Result<Self, RedisCatalogError> {
        let url = reqwest::Url::parse(value)
            .map_err(|error| RedisCatalogError::Url(error.to_string()))?;
        let tls = match url.scheme() {
            "redis" => false,
            "rediss" => true,
            scheme => {
                return Err(RedisCatalogError::Url(format!(
                    "scheme {scheme} is not redis or rediss"
                )));
            }
        };
        let host = url
            .host_str()
            .filter(|host| !host.is_empty())
            .ok_or_else(|| RedisCatalogError::Url("host is missing".to_string()))?
            .to_string();
        let username = Some(url.username())
            .filter(|username| !username.is_empty())
            .map(str::to_string);
        let password = url
            .password()
            .filter(|password| !password.is_empty())
            .map(str::to_string);
        if username.is_some() && password.is_none() {
            return Err(RedisCatalogError::Url(
                "a username requires a password".to_string(),
            ));
        }
        let path = url.path().trim_matches('/');
        let database = if path.is_empty() {
            None
        } else {
            Some(path.parse().map_err(|_| {
                RedisCatalogError::Url("database in the path must be a number".to_string())
            })?)
        };
        if url.query().is_some() || url.fragment().is_some() {
            return Err(RedisCatalogError::Url(
                "query strings and fragments are not supported".to_string(),
            ));
        }
        Ok(Self {
            tls,
            host,
            port: url.port().unwrap_or(6379),
            username,
            password,
            database,
        })
    }
}

trait AsyncStream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> AsyncStream for T {}

#[derive(Debug, PartialEq, Eq)]
enum Reply {
    Simple(String),
    Integer(i64),
    Bulk(Option<Vec<u8>>),
    Array(Option<Vec<Reply>>),
    Null,
    Boolean(bool),
}

impl Reply {
    fn kind(&self) -> &'static str {
        match self {
            Self::Simple(_) => "a simple string",
            Self::Integer(_) => "an integer",
            Self::Bulk(_) => "a bulk string",
            Self::Array(_) => "an array",
            Self::Null => "null",
            Self::Boolean(_) => "a boolean",
        }
    }
}

fn expect_ok(reply: &Reply, command: &str) -> std::result::Result<(), RedisCatalogError> {
    match reply {
        Reply::Simple(message) if message == "OK" => Ok(()),
        Reply::Simple(message) => Err(RedisCatalogError::Protocol(format!(
            "{command} returned {message:?}"
        ))),
        other => Err(RedisCatalogError::Protocol(format!(
            "{command} returned {}",
            other.kind()
        ))),
    }
}

async fn command(
    stream: &mut (impl AsyncRead + AsyncWrite + Unpin),
    parts: &[&str],
) -> std::result::Result<Reply, RedisCatalogError> {
    let mut request = format!("*{}\r\n", parts.len()).into_bytes();
    for part in parts {
        request.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
        request.extend_from_slice(part.as_bytes());
        request.extend_from_slice(b"\r\n");
    }
    stream.write_all(&request).await?;
    stream.flush().await?;
    read_reply(stream).await
}

async fn read_reply(
    stream: &mut (impl AsyncRead + Unpin),
) -> std::result::Result<Reply, RedisCatalogError> {
    let line = read_line(stream).await?;
    let (kind, rest) = line
        .split_at_checked(1)
        .ok_or_else(|| RedisCatalogError::Protocol("empty reply".to_string()))?;
    match kind {
        "+" => Ok(Reply::Simple(rest.to_string())),
        "-" => Err(RedisCatalogError::Server(rest.to_string())),
        ":" => Ok(Reply::Integer(rest.parse().map_err(|_| {
            RedisCatalogError::Protocol("invalid integer reply".to_string())
        })?)),
        "$" => {
            let length: i64 = rest
                .parse()
                .map_err(|_| RedisCatalogError::Protocol("invalid bulk length".to_string()))?;
            if length < 0 {
                return Ok(Reply::Bulk(None));
            }
            let length = usize::try_from(length)
                .map_err(|_| RedisCatalogError::Protocol("bulk length is too large".to_string()))?;
            if length > MAX_BULK_BYTES {
                return Err(RedisCatalogError::Protocol(
                    "bulk reply exceeds the size limit".to_string(),
                ));
            }
            let mut value = vec![0; length];
            stream.read_exact(&mut value).await?;
            let mut crlf = [0; 2];
            stream.read_exact(&mut crlf).await?;
            if crlf != *b"\r\n" {
                return Err(RedisCatalogError::Protocol(
                    "bulk reply did not end with CRLF".to_string(),
                ));
            }
            Ok(Reply::Bulk(Some(value)))
        }
        "*" => {
            let length: i64 = rest
                .parse()
                .map_err(|_| RedisCatalogError::Protocol("invalid array length".to_string()))?;
            if length < 0 {
                return Ok(Reply::Array(None));
            }
            let length = usize::try_from(length).map_err(|_| {
                RedisCatalogError::Protocol("array length is too large".to_string())
            })?;
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(Box::pin(read_reply(stream)).await?);
            }
            Ok(Reply::Array(Some(values)))
        }
        "_" => Ok(Reply::Null),
        "#" => match rest {
            "t" => Ok(Reply::Boolean(true)),
            "f" => Ok(Reply::Boolean(false)),
            _ => Err(RedisCatalogError::Protocol(
                "invalid boolean reply".to_string(),
            )),
        },
        other => Err(RedisCatalogError::Protocol(format!(
            "unknown reply type {other:?}"
        ))),
    }
}

async fn read_line(
    stream: &mut (impl AsyncRead + Unpin),
) -> std::result::Result<String, RedisCatalogError> {
    let mut line = Vec::new();
    let mut byte = [0; 1];
    loop {
        let read = stream.read(&mut byte).await?;
        if read == 0 {
            return Err(RedisCatalogError::Protocol(
                "connection closed before a complete reply".to_string(),
            ));
        }
        if byte[0] == b'\r' {
            let mut lf = [0; 1];
            stream.read_exact(&mut lf).await?;
            if lf[0] != b'\n' {
                return Err(RedisCatalogError::Protocol(
                    "reply line ended with a bare carriage return".to_string(),
                ));
            }
            break;
        }
        line.push(byte[0]);
        if line.len() > MAX_LINE_BYTES {
            return Err(RedisCatalogError::Protocol(
                "reply line exceeds the size limit".to_string(),
            ));
        }
    }
    String::from_utf8(line)
        .map_err(|_| RedisCatalogError::Protocol("reply line is not UTF-8".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog_json() -> String {
        serde_json::json!([{
            "subtask_id": "edit-code",
            "subtask_name": "Edit code",
            "definition": "Change existing source files.",
            "candidates": [
                {"model_id": "model/cheap", "cost_per_million_tokens": 0.5},
                {"model_id": "model/pricy", "cost_per_million_tokens": 9.0}
            ],
            "skill": {"version": "v1", "template": "Minimal patch."}
        }])
        .to_string()
    }

    async fn serve_once(reply: impl FnOnce() -> Vec<u8> + Send + 'static) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut seen = Vec::new();
            loop {
                let request = read_reply(&mut socket).await.expect("command");
                let Reply::Array(Some(parts)) = request else {
                    panic!("expected a command array");
                };
                let parts: Vec<String> = parts
                    .into_iter()
                    .map(|part| match part {
                        Reply::Bulk(Some(bytes)) => String::from_utf8(bytes).expect("utf8"),
                        other => panic!("expected bulk command part, got {other:?}"),
                    })
                    .collect();
                seen.push(parts.join(" "));
                if seen
                    .last()
                    .is_some_and(|command| command.starts_with("GET "))
                {
                    socket.write_all(&reply()).await.expect("write");
                    break;
                }
                socket.write_all(b"+OK\r\n").await.expect("write");
            }
            drop(seen);
        });
        format!("redis://{address}")
    }

    #[tokio::test]
    async fn reads_the_subtask_catalog() {
        let json = catalog_json();
        let url = serve_once(move || {
            let json = json.as_bytes();
            let mut reply = format!("${}\r\n", json.len()).into_bytes();
            reply.extend_from_slice(json);
            reply.extend_from_slice(b"\r\n");
            reply
        })
        .await;
        let catalog = RedisSubtaskCatalog::new(&url, "rsi:subtasks", Duration::from_secs(2))
            .expect("catalog");
        let entries = catalog.fetch_entries().await.expect("catalog");
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.subtask_id, "edit-code");
        assert_eq!(entry.subtask_name, "Edit code");
        assert_eq!(entry.definition, "Change existing source files.");
        assert_eq!(entry.candidates[0].model_id, "model/cheap");
        assert_eq!(
            entry.skill.as_ref().map(|skill| skill.version.as_str()),
            Some("v1")
        );
    }

    #[tokio::test]
    async fn a_missing_key_is_an_empty_catalog() {
        let url = serve_once(|| b"$-1\r\n".to_vec()).await;
        let catalog = RedisSubtaskCatalog::new(&url, "rsi:subtasks", Duration::from_secs(2))
            .expect("catalog");
        assert!(catalog.fetch_entries().await.expect("catalog").is_empty());
    }

    #[tokio::test]
    async fn invalid_json_is_an_error() {
        let url = serve_once(|| b"$3\r\nbad\r\n".to_vec()).await;
        let catalog = RedisSubtaskCatalog::new(&url, "rsi:subtasks", Duration::from_secs(2))
            .expect("catalog");
        assert!(catalog.fetch_entries().await.is_err());
    }

    #[tokio::test]
    async fn authenticates_and_selects_a_database() {
        let json = catalog_json();
        let reply = {
            let json = json.clone();
            move || {
                let json = json.as_bytes();
                let mut reply = format!("${}\r\n", json.len()).into_bytes();
                reply.extend_from_slice(json);
                reply.extend_from_slice(b"\r\n");
                reply
            }
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut commands = Vec::new();
            for _ in 0..3 {
                let request = read_reply(&mut socket).await.expect("command");
                let Reply::Array(Some(parts)) = request else {
                    panic!("expected a command array");
                };
                let parts: Vec<String> = parts
                    .into_iter()
                    .map(|part| match part {
                        Reply::Bulk(Some(bytes)) => String::from_utf8(bytes).expect("utf8"),
                        other => panic!("expected bulk command part, got {other:?}"),
                    })
                    .collect();
                commands.push(parts.join(" "));
                if commands.len() == 3 {
                    socket.write_all(&reply()).await.expect("write");
                } else {
                    socket.write_all(b"+OK\r\n").await.expect("write");
                }
            }
            commands
        });
        let catalog = RedisSubtaskCatalog::new(
            &format!("redis://user:secret@{address}/2"),
            "rsi:subtasks",
            Duration::from_secs(2),
        )
        .expect("catalog");
        assert!(!catalog.fetch_entries().await.expect("catalog").is_empty());
        let commands = server.await.expect("server");
        assert_eq!(commands[0], "AUTH user secret");
        assert_eq!(commands[1], "SELECT 2");
        assert_eq!(commands[2], "GET rsi:subtasks");
    }

    #[tokio::test]
    async fn an_unavailable_catalog_reports_an_error() {
        let catalog = UnavailableSubtaskCatalog::new("RSI_REDIS_URL is not set");
        let error = catalog
            .entries()
            .await
            .expect_err("unavailable catalog should fail");
        assert!(error.to_string().contains("RSI_REDIS_URL is not set"));
    }

    #[test]
    fn parses_redis_urls_without_exposing_credentials() {
        let url = RedisUrl::parse("rediss://user:secret@example.test:6380/3").expect("url");
        assert!(url.tls);
        assert_eq!(url.host, "example.test");
        assert_eq!(url.port, 6380);
        assert_eq!(url.username.as_deref(), Some("user"));
        assert_eq!(url.password.as_deref(), Some("secret"));
        assert_eq!(url.database, Some(3));

        assert!(RedisUrl::parse("http://example.test").is_err());
        assert!(RedisUrl::parse("redis://user@example.test").is_err());
        assert!(RedisUrl::parse("redis://example.test/not-a-number").is_err());
    }
}
