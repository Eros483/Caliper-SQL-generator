select
    intrv_type,
    abbr,
    org_id,
    label,
    tooltip,
    is_compound::varchar = '1' as is_compound,
    compound_abbrs,
    cost_pre,
    cost_delta,
    score_impact,
    gap_closure_impact,
    baseline_gap_closure
from {{ source('mysql', 'intervention_type') }}
