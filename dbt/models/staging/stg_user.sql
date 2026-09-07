select
    lower(hex(user_id)) as user_id,
    first_name,
    last_name,
    org_id,
    is_admin::varchar = '1' as is_admin
from {{ source('mysql', 'user') }}
