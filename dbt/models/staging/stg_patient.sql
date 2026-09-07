select
    lower(hex(patient_id)) as patient_id,
    first_name,
    last_name,
    middle_name,
    date_of_birth,
    sex::varchar as sex,
    lower(hex(patient_coordinator)) as patient_coordinator,
    race,
    ethnicity,
    language
from {{ source('mysql', 'patient') }}
