DROP FUNCTION IF EXISTS vectorize."rag";

CREATE  FUNCTION vectorize."rag"(
	"job_name" TEXT, /* &str */
	"query" TEXT, /* &str */
	"chat_model" TEXT DEFAULT 'openai/gpt-4o-mini', /* alloc::string::String */
	"task" TEXT DEFAULT 'question_answer', /* alloc::string::String */
	"api_key" TEXT DEFAULT NULL, /* core::option::Option<alloc::string::String> */
	"num_context" INT DEFAULT 2, /* i32 */
	"force_trim" bool DEFAULT false /* bool */
) RETURNS TABLE (
	"chat_results" jsonb  /* pgrx::datum::json::JsonB */
)
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'rag_wrapper';

DROP FUNCTION vectorize."table";
-- src/api.rs:94
-- vectorize::api::table
CREATE  FUNCTION vectorize."table"(
	"relation" TEXT, /* &str */
	"columns" TEXT[], /* alloc::vec::Vec<alloc::string::String> */
	"job_name" TEXT, /* &str */
	"primary_key" TEXT, /* &str */
	"schema" TEXT DEFAULT 'public', /* &str */
	"update_col" TEXT DEFAULT 'last_updated_at', /* alloc::string::String */
	"index_dist_type" IndexDist DEFAULT 'pgv_hnsw_cosine', /* vectorize::types::IndexDist */
	"transformer" TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2', /* &str */
	"table_method" TableMethod DEFAULT 'join', /* vectorize::types::TableMethod */
	"schedule" TEXT DEFAULT '* * * * *' /* &str */
) RETURNS TEXT /* core::result::Result<alloc::string::String, anyhow::Error> */
STRICT
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'table_wrapper';
/* </end connected objects> */

DROP FUNCTION IF EXISTS vectorize."table_from";
-- src/api.rs:380
-- vectorize::api::table_from
CREATE  FUNCTION vectorize."table_from"(
	"relation" TEXT, /* &str */
	"columns" TEXT[], /* alloc::vec::Vec<alloc::string::String> */
	"job_name" TEXT, /* &str */
	"primary_key" TEXT, /* &str */
	"src_table" TEXT, /* &str */
	"src_primary_key" TEXT, /* &str */
	"src_embeddings_col" TEXT, /* &str */
	"schema" TEXT DEFAULT 'public', /* &str */
	"update_col" TEXT DEFAULT 'last_updated_at', /* alloc::string::String */
	"index_dist_type" IndexDist DEFAULT 'pgv_hnsw_cosine', /* vectorize::types::IndexDist */
	"transformer" TEXT DEFAULT 'sentence-transformers/all-MiniLM-L6-v2', /* &str */
	"table_method" TableMethod DEFAULT 'join', /* vectorize::types::TableMethod */
	"schedule" TEXT DEFAULT '* * * * *' /* &str */
) RETURNS TEXT /* core::result::Result<alloc::string::String, anyhow::Error> */
STRICT
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'table_from_wrapper';

-- Rename 'table' key to 'relation' in the params JSONB column
UPDATE vectorize.job
SET params = jsonb_set(
    params - 'table',  -- Remove old 'table' key
    '{relation}', 
    params->'table',  -- Copy value to new 'relation' key
    true  -- Ensure key exists
)
WHERE params ? 'table';

-- Update the function to reference "relation" instead of "table"
CREATE OR REPLACE FUNCTION vectorize.handle_table_drop()
RETURNS event_trigger AS $$
DECLARE
    obj RECORD;
    schema_name TEXT;
    relation_name TEXT;
BEGIN
    FOR obj IN SELECT * FROM pg_event_trigger_dropped_objects() LOOP
        IF obj.object_type = 'table' THEN
            schema_name := split_part(obj.object_identity, '.', 1);  
            relation_name := split_part(obj.object_identity, '.', 2);  
            
            -- Perform cleanup: delete the associated job from the vectorize.job table
            DELETE FROM vectorize.job
            WHERE params ->> 'relation' = relation_name
            AND params ->> 'schema' = schema_name;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;