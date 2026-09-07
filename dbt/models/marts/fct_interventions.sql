-- One row per patient-intervention event, pre-joined for cost analysis.
-- ponytail: no cost_actual column exists upstream; actual = baseline + delta.
select
    s.patient_id,
    s.intrv_type,
    t.label as intervention_label,
    s.service_date,
    s.source,
    coalesce(t.cost_pre, 0) + coalesce(t.cost_delta, 0) as cost_actual,
    o.display_name as organization_name
from {{ ref('stg_intervention_service') }} s
left join {{ ref('stg_intervention_type') }} t on t.intrv_type = s.intrv_type
left join {{ ref('stg_organization') }} o on o.org_id = t.org_id
