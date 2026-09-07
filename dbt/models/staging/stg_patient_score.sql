select
    patient_score_id,
    lower(hex(patient_id)) as patient_id,
    calculation_time,
    comp_score,
    comp_confidence,
    impactability,
    risk_contributors_count,
    provided_interventions,
    needed_interventions_count,
    hcc_score,
    hccs,
    sdoh_score,
    service_date_end,
    quality_score,
    er_visits,
    ip_visits
from {{ source('mysql', 'patient_score') }}
