select
    org_id,
    lower(hex(org_guid)) as org_guid,
    full_name,
    display_name,
    cohort_name,
    patient_name
from {{ source('mysql', 'organization') }}
