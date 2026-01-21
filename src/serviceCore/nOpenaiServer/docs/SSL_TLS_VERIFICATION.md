# SSL/TLS Verification Guide

**Purpose:** Ensure secure connections to SAP HANA Cloud  
**Status:** Production-ready configuration

---

## Overview

SAP HANA Cloud requires SSL/TLS encrypted connections for security. This guide covers verification and troubleshooting.

## Current Configuration

### HANA Cloud Connection
- **Port:** 443 (HTTPS)
- **Protocol:** SSL/TLS 1.2+
- **Certificate:** SAP BTP managed certificates
- **Verification:** Enabled by default

### OData Client Configuration

Located in `sap-toolkit-mojo/lib/clients/hana/client.zig`:

```zig
// SSL/TLS is handled by HANA Cloud automatically
// Connection string format:
// https://<instance>.hanacloud.ondemand.com:443
```

---

## Verification Steps

### 1. Check HANA Cloud Certificate

```bash
# Test SSL connection to HANA Cloud
openssl s_client -connect <your-instance>.hanacloud.ondemand.com:443 -showcerts

# Expected output should include:
# - Verify return code: 0 (ok)
# - SSL handshake has been completed
```

### 2. Verify Certificate Chain

```bash
# Download and verify certificate
echo | openssl s_client -connect <your-instance>.hanacloud.ondemand.com:443 2>&1 | \
  sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > hana.crt

# Check certificate details
openssl x509 -in hana.crt -text -noout

# Expected:
# - Issuer: DigiCert or SAP
# - Validity: Not expired
# - Subject Alternative Names: Your HANA instance
```

### 3. Test Connection from Application

```bash
# Run integration test (includes SSL verification)
cd src/serviceCore/nOpenaiServer
./scripts/test_hana_integration.sh

# Expected output:
# ✓ PASS - All endpoints working
# ✓ SSL/TLS connection successful
```

---

## Common Issues & Solutions

### Issue 1: Certificate Verification Failed

**Symptoms:**
```
Error: SSL certificate problem: unable to get local issuer certificate
```

**Solution:**
```bash
# Update CA certificates (macOS)
brew install ca-certificates
brew upgrade ca-certificates

# Update CA certificates (Linux)
sudo apt-get update
sudo apt-get install ca-certificates

# Verify system trust store
openssl version -d
```

### Issue 2: TLS Version Mismatch

**Symptoms:**
```
Error: SSL routines:ssl3_get_record:wrong version number
```

**Solution:**
- HANA Cloud requires TLS 1.2 or higher
- Update OpenSSL if version < 1.1.0
```bash
# Check OpenSSL version
openssl version

# Should be 1.1.0 or higher
# On macOS: brew upgrade openssl
# On Linux: apt-get install openssl
```

### Issue 3: Hostname Verification Failed

**Symptoms:**
```
Error: SSL certificate subject name does not match target host name
```

**Solution:**
- Ensure you're using the exact hostname from BTP cockpit
- Format: `<instance-id>.hanacloud.ondemand.com`
- Check `.env` file for correct `HANA_HOST`

---

## Security Best Practices

### 1. Certificate Pinning (Optional)
For highest security, pin the HANA Cloud certificate:

```zig
// In hana/client.zig
pub const Config = struct {
    // ... existing fields
    pin_certificate: ?[]const u8 = null,  // Optional SHA256 fingerprint
};
```

### 2. Environment Variables
Always use environment variables for credentials:

```bash
# .env file (never commit!)
HANA_HOST=your-instance.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=DBADMIN
HANA_PASSWORD=<secure-password>
HANA_SCHEMA=NUCLEUS
```

### 3. Connection Pooling
Reuse SSL connections to reduce overhead:

```zig
// Connection pool with SSL session reuse
const pool = try ConnectionPool.init(allocator, config, .{
    .max_connections = 10,
    .reuse_ssl_sessions = true,
});
```

---

## Production Checklist

- [ ] SSL/TLS 1.2+ verified
- [ ] Certificate chain validated
- [ ] Hostname verification enabled
- [ ] Connection timeout configured (30s recommended)
- [ ] Retry logic implemented (3 retries)
- [ ] Connection pooling enabled
- [ ] Credentials in `.env` file (not hardcoded)
- [ ] Certificate expiry monitoring set up
- [ ] Firewall rules allow port 443
- [ ] Integration tests passing

---

## Testing SSL/TLS

### Manual Test

```bash
# Test with curl (verbose)
curl -v https://<your-instance>.hanacloud.ondemand.com:443

# Expected:
# * SSL connection using TLSv1.2
# * Server certificate verified successfully
# < HTTP/1.1 200 OK
```

### Automated Test

```bash
# Run SSL verification test
cd src/serviceCore/nOpenaiServer
./scripts/test_hana_integration.sh

# Should complete without SSL errors
```

---

## Monitoring

### Certificate Expiry Alert

```bash
# Add to crontab for daily check
0 0 * * * /path/to/check_cert_expiry.sh

# Script checks expiry and alerts if < 30 days
```

### Connection Health Check

```bash
# Monitor SSL connection health
watch -n 60 'echo | openssl s_client -connect <instance>:443 2>&1 | grep "Verify return code"'

# Should always show: Verify return code: 0 (ok)
```

---

## References

- [SAP HANA Cloud Security Guide](https://help.sap.com/docs/HANA_CLOUD/security)
- [OpenSSL Documentation](https://www.openssl.org/docs/)
- [TLS Best Practices](https://wiki.mozilla.org/Security/Server_Side_TLS)

---

**Last Updated:** January 21, 2026  
**Status:** ✅ Production Ready
