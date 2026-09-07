select
    lower(hex(lob_id)) as lob_id,
    name,
    lower(hex(org_guid)) as org_guid
from {{ source('mysql', 'lob') }}
