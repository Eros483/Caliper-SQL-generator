{% macro attach_mysql() %}
  {# ponytail: creds mirror backend defaults (root/empty pw); override via DB_* env vars #}
  {% set conn = 'host=' ~ env_var('DB_HOST', '127.0.0.1') ~ ' user=' ~ env_var('DB_USER', 'root') ~ ' port=3306 database=' ~ env_var('DB_NAME', 'fhs_coredb_local') %}
  {% if env_var('DB_PASSWORD', '') != '' %}{% set conn = conn ~ ' password=' ~ env_var('DB_PASSWORD', '') %}{% endif %}
  LOAD mysql;
  ATTACH '{{ conn }}' AS mysql_source (TYPE mysql, READ_ONLY);
{% endmacro %}
