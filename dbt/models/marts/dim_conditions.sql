-- One row per patient-condition pair, from contributor findings.
-- ponytail: category derived from org-set flags + a small mental-health label list.
select distinct
    ci.patient_id,
    ct.label as condition_name,
    case
        when lower(ct.label) like '%anxiety%' or lower(ct.label) like '%depress%' or lower(ct.label) like '%bipolar%' then 'Mental Health'
        when ct.is_sdoh then 'SDOH'
        else 'Chronic Disease'
    end as condition_category
from {{ ref('stg_contributor_individual') }} ci
join {{ ref('stg_contributor_type') }} ct on ct.contr_type = ci.contr_type
