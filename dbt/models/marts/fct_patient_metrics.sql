-- One row per patient: demographics + insurance/org + latest risk scores.
with latest_metric as (
    select *, row_number() over (partition by patient_id order by start_date desc, end_date desc nulls last) as rn
    from {{ ref('stg_map_patient_metrics') }}
),
latest_score as (
    select *, row_number() over (partition by patient_id order by calculation_time desc, service_date_end desc) as rn
    from {{ ref('stg_patient_score') }}
)
select
    p.patient_id,
    p.first_name,
    p.last_name,
    p.date_of_birth,
    p.sex,
    p.race,
    p.ethnicity,
    p.language,
    l.name as insurance_name,
    o.display_name as organization_name,
    s.calculation_time as score_time,
    s.comp_score,
    s.impactability,
    s.hcc_score,
    s.sdoh_score,
    s.quality_score,
    s.er_visits,
    s.ip_visits
from {{ ref('stg_patient') }} p
left join latest_metric m on m.patient_id = p.patient_id and m.rn = 1
left join {{ ref('stg_lob') }} l on l.lob_id = m.lob_id
left join {{ ref('stg_organization') }} o on o.org_guid = l.org_guid
left join latest_score s on s.patient_id = p.patient_id and s.rn = 1
