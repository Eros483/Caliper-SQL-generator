-- Slim patient dimension for listing/filtering.
with latest_metric as (
    select *, row_number() over (partition by patient_id order by start_date desc, end_date desc nulls last) as rn
    from {{ ref('stg_map_patient_metrics') }}
)
select
    p.patient_id,
    p.first_name,
    p.last_name,
    p.date_of_birth,
    p.sex,
    l.name as insurance_name
from {{ ref('stg_patient') }} p
left join latest_metric m on m.patient_id = p.patient_id and m.rn = 1
left join {{ ref('stg_lob') }} l on l.lob_id = m.lob_id
