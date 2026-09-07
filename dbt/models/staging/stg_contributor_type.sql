select
    contr_type,
    abbr,
    org_id,
    label,
    tooltip,
    is_diag::varchar = '1' as is_diag,
    is_sdoh::varchar = '1' as is_sdoh,
    coefficient,
    coefficient_quality,
    coefficient_qip
from {{ source('mysql', 'contributor_type') }}
