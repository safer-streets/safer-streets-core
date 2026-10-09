# Azure Blob Storage access lockdown — 2026-10-09

## Problem

Blobs in the `saferstreets` storage account could be downloaded by anyone, with no credentials, e.g.

```sh
curl https://saferstreets.blob.core.windows.net/phase2/index.parquet --output test.parquet
```

Requirement: no anonymous access, and two tiers of restricted access, each given as a connection string
that works with both pandas and DuckDB:

- **read-only**, for the specific users who need to read the data
- **read-write**, for a smaller group who also write the data

The existing `AZURE_STORAGE_CONNSTR` variable must keep working for code that reads data.

## Connection strings

Since these changes, `safer-streets/.env` holds one connection string per tier:

| Variable                      | Tier       | Contents                                     | Access                                           | Used for |
|-------------------------------|------------|----------------------------------------------|--------------------------------------------------|----------|
| `AZURE_STORAGE_CONNSTR`       | read-only  | SAS (`readers` policy on `phase2`)           | read and list `phase2` only, expires 2028-03-31  | all code that reads data (`safer_streets_core`, eda notebooks, peer-hex-explorer, etc.) |
| `AZURE_STORAGE_ADMIN_CONNSTR` | read-write | account key                                  | full read, write and delete on the whole account | writing data, and admin: SAS policies and tokens, container settings |

Both work with pandas (`storage_options={"connection_string": ...}`) and DuckDB (`CREATE SECRET` or
`SET azure_storage_connection_string`); see [Usage](#usage).

When the work below was done, the account-key string was still called `AZURE_STORAGE_CONNSTR`. The
commands in this document now use `AZURE_STORAGE_ADMIN_CONNSTR` wherever they need the account key.

Code that reads data needs no changes, because it still reads `AZURE_STORAGE_CONNSTR`. Readers get
the same SAS string under the same name, so their code matches ours.

The GitHub Actions secrets are separate from `.env` and were **not** changed. In particular,
`site/.github/workflows/deploy.yml` uses its `AZURE_STORAGE_CONNSTR` secret to mint per-blob SAS
tokens, so that secret must stay the account-key string.

## Cause

Both containers had their public access level set to `blob`, which allows anonymous read of any
blob whose URL is known:

| Container            | Public access (before) |
|----------------------|------------------------|
| `phase2`             | `blob`                 |
| `safer-streets-data` | `blob`                 |

## Change made

Anonymous public access was turned off on both containers:

```sh
az storage container set-permission -n phase2             --public-access off --connection-string "$AZURE_STORAGE_ADMIN_CONNSTR"
az storage container set-permission -n safer-streets-data --public-access off --connection-string "$AZURE_STORAGE_ADMIN_CONNSTR"
```

At this stage nothing else was changed (the account-level setting below came later): account keys, CORS rules, role assignments and
blob contents were not modified.

### How access was obtained

- The `az login` session had expired (University of Leeds conditional access requires sign-in
  every 30 days), so Azure AD / control-plane access was unavailable.
- `AZURE_STORAGE_CONNSTR` (now `AZURE_STORAGE_ADMIN_CONNSTR`) was loaded from `safer-streets/.env`.
  It is an account-key connection string (`DefaultEndpointsProtocol=https;AccountName=saferstreets;AccountKey=...`).
- The account key is allowed to set container ACLs via the blob data-plane API, which is what
  `az storage container set-permission` uses.

## Verification

| Check                                                         | Result                          |
|---------------------------------------------------------------|---------------------------------|
| Container public access level (both containers)               | none (private)                  |
| Anonymous `curl` of `phase2/index.parquet`                    | HTTP 404 (Azure's response to anonymous requests for private blobs) |
| `az storage blob show` on `phase2/index.parquet` with the account-key string | succeeds (11,989 bytes) |

### Impact on existing consumers

None of these rely on anonymous access, so none should be affected. This is based on reading the
code; they were not run end to end:

- **prototype-explorer-app**: uses short-lived, read-only SAS tokens.
- **safer-streets-tooling / safer-streets-core**: authenticate with a service principal
  (account URL from `SAFER_STREETS_BLOB_STORAGE`). `data sync` was later checked after both the
  container and account-level changes (see [`data sync` check](#data-sync-check)).
- **safer-streets-eda** DuckDB `az://phase2/...` reads: authenticate through a configured
  DuckDB Azure secret.

The account key connection string is unaffected by container public access settings.

## Rollback

```sh
az storage container set-permission -n <container> --public-access blob --connection-string "$AZURE_STORAGE_ADMIN_CONNSTR"
```

### Account-level setting

Public access was then also disallowed at account level, so no container can be made public again.
This is a control-plane setting, so it needs an Azure AD login (the account key cannot change it),
and the subscription must be set to the one that holds the storage account:

```sh
az account set --subscription <subscription-id>
az storage account update -n saferstreets -g <resource-group> --allow-blob-public-access false
```

Verified state: `allowBlobPublicAccess = false`, `allowSharedKeyAccess = true`. Shared-key access
must stay enabled. Disabling it would break both connection strings: `AZURE_STORAGE_ADMIN_CONNSTR`
uses the account key directly, and the read-only SAS in `AZURE_STORAGE_CONNSTR` is signed with it.

Rechecked later on 2026-10-09 with `az storage account show`, which still reported
`allowBlobPublicAccess = false`. To check it again:

```sh
az storage account show -n saferstreets -g <resource-group> \
  --subscription <subscription-id> \
  --query "{allowBlobPublicAccess:allowBlobPublicAccess, allowSharedKeyAccess:allowSharedKeyAccess}"
```

## Read-only SAS connection string for named readers

A read-only SAS, tied to a stored access policy called `readers` on `phase2`, was created with the
account-key connection string:

```sh
az storage container policy create -c phase2 -n readers --permissions rl --expiry 2028-03-31 --connection-string "$AZURE_STORAGE_ADMIN_CONNSTR"
az storage container generate-sas -n phase2 --policy-name readers --https-only --connection-string "$AZURE_STORAGE_ADMIN_CONNSTR" -o tsv
```

The second command prints a bare token (`spr=https&sv=...&si=readers&sr=c&sig=...`). To turn it into a
connection string, add the account details in front of it:

```text
DefaultEndpointsProtocol=https;AccountName=saferstreets;EndpointSuffix=core.windows.net;SharedAccessSignature=<token>
```

Use this form, which includes `AccountName`. The shorter
`BlobEndpoint=https://saferstreets.blob.core.windows.net;SharedAccessSignature=<token>` works with
pandas and with a DuckDB `CREATE SECRET`, but DuckDB's `SET azure_storage_connection_string` rejects
it ("A invalid connection string has been provided"). `safer_streets_core.database` uses that `SET`.

This string is what `AZURE_STORAGE_CONNSTR` holds in `.env` (see [Connection strings](#connection-strings)).
Give readers the same string under the same name, so their code is the same as ours.

### Usage

pandas (needs `adlfs`):

```python
import os
import pandas as pd

df = pd.read_parquet(
    "az://phase2/index.parquet",
    storage_options={"connection_string": os.environ["AZURE_STORAGE_CONNSTR"]},
)
```

DuckDB:

```python
import os
import duckdb

con = duckdb.connect()
con.sql(f"CREATE SECRET (TYPE azure, CONNECTION_STRING '{os.environ['AZURE_STORAGE_CONNSTR']}')")
df = con.sql("SELECT * FROM read_parquet('az://phase2/index.parquet')").df()
```

`safer_streets_core.database.duckdb_connector(azure=True)` also works unchanged with the SAS string.

The read-write tier uses the same code with `AZURE_STORAGE_ADMIN_CONNSTR` in place of
`AZURE_STORAGE_CONNSTR`.

### Verification (2026-10-09)

| Check                                                              | Result                     |
|--------------------------------------------------------------------|----------------------------|
| pandas `read_parquet("az://phase2/index.parquet")`                 | works (69 × 9)             |
| DuckDB with `CREATE SECRET`                                        | works                      |
| DuckDB with `SET azure_storage_connection_string` + curl transport | works                      |
| List blobs in `phase2`                                             | works (56 blobs)           |
| Upload to `phase2`                                                 | refused                    |
| Access `safer-streets-data`                                        | refused                    |

### Revoking

Every copy of the SAS shares the `readers` policy, so deleting the policy revokes them all at once.
This needs the account-key string, because the SAS string can't manage policies:

```sh
az storage container policy delete -c phase2 -n readers --connection-string "$AZURE_STORAGE_ADMIN_CONNSTR"
```

To be able to revoke one person at a time, give each person their own policy and SAS. A container
can have at most 5 policies.

### Keep the token out of git

Store the connection string in `.env` (which is git-ignored), never in a tracked file.

## `data sync` check

`data sync` (safer-streets-tooling) doesn't use `AZURE_STORAGE_CONNSTR`. It uses
`DefaultAzureCredential` (the service principal in `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` /
`AZURE_CLIENT_SECRET`), which signs in through Azure AD. Neither the container change nor the
account-level change affects that.

Checked on 2026-10-09 after both changes. The test used `AzureBlobStorage(blob_storage_url(), "phase2",
readonly=False)`, the same client `sync` uses, with `AZURE_STORAGE_CONNSTR` unset:

| Operation used by `sync`                 | Result                        |
|------------------------------------------|-------------------------------|
| list blobs                               | works (56 blobs)              |
| read `index.parquet`                     | works (11,989 bytes)          |
| `write_file` (upload, with metadata)     | works                         |
| blob properties (`metadata`)             | works                         |
| delete (clean-up of the test blob)       | works; test blob removed      |

No changes are needed for `data sync`.

## Still to do

These need an Azure AD login (`az login --tenant <tenant-id>`) with the
subscription set as above.

1. **Grant access to specific users**, using either of these:
   - **Azure AD role** (users in the Leeds tenant). Users then download with
     `az storage blob download --auth-mode login ...`:

     ```sh
     az role assignment create --assignee <user>@leeds.ac.uk --role "Storage Blob Data Reader" \
       --scope /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.Storage/storageAccounts/saferstreets/blobServices/default/containers/phase2
     ```

   - **Read-only SAS connection string** (any user, including external): done for `phase2`. See
     [Read-only SAS connection string for named readers](#read-only-sas-connection-string-for-named-readers).
     What's left is sending it to each reader securely, not by email in plain text.

2. **Consider rotating the account key** if it might have been shared more widely than intended.
   The key gives full access to the account. Rotating it means updating `AZURE_STORAGE_ADMIN_CONNSTR`
   in `.env` and the site repo's `AZURE_STORAGE_CONNSTR` GitHub Actions secret (plus peer-hex-explorer's,
   if that one also holds the account key).
   Rotating the key the SAS was signed with also invalidates the read-only SAS, so it would need
   regenerating and re-sending to readers.
