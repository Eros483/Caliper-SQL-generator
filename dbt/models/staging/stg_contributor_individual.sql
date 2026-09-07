select
    lower(hex(patient_id)) as patient_id,
    contr_type,
    identified_on_date,
    src_label
from {{ source('mysql', 'contributor_individual') }}
