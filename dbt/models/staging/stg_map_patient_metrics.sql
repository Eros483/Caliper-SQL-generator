select
    lower(hex(patient_id)) as patient_id,
    lower(hex(group_id)) as group_id,
    lower(hex(lob_id)) as lob_id,
    lower(hex(plan_id)) as plan_id,
    lower(hex(program_id)) as program_id,
    start_date,
    end_date
from {{ source('mysql', 'map_patient_metrics') }}
