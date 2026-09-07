select
    lower(hex(patient_id)) as patient_id,
    intrv_type,
    service_date,
    source
from {{ source('mysql', 'intervention_service') }}
