-- ============================================================
-- Spending Forecast App — Supabase schema
-- Run once in the Supabase SQL editor.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- merchant_overrides
CREATE TABLE IF NOT EXISTS merchant_overrides (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_key     text UNIQUE NOT NULL,
    merchant_original text NOT NULL,
    category         text NOT NULL,
    source           text NOT NULL DEFAULT 'manual'
                         CHECK (source IN ('manual','llm_accepted','llm_auto')),
    approved_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_overrides_key ON merchant_overrides (merchant_key);

-- saving_goals
CREATE TABLE IF NOT EXISTS saving_goals (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    person                  text NOT NULL,
    effective_month         date NOT NULL,
    monthly_savings_target  numeric(12,2) NOT NULL DEFAULT 0,
    category_caps           jsonb DEFAULT '{}'::jsonb,
    updated_at              timestamptz NOT NULL DEFAULT now(),
    UNIQUE (person, effective_month)
);
CREATE INDEX IF NOT EXISTS idx_goals_person ON saving_goals (person, effective_month DESC);

-- llm_cache (embedding nullable — works without sentence-transformers)
CREATE TABLE IF NOT EXISTS llm_cache (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key_hash   text UNIQUE NOT NULL,
    embedding        vector(384),
    report_markdown  text NOT NULL,
    person           text NOT NULL,
    model_used       text NOT NULL,
    date_range_start date,
    date_range_end   date,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cache_person ON llm_cache (person, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cache_key    ON llm_cache (cache_key_hash);

-- csv_files
CREATE TABLE IF NOT EXISTS csv_files (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    person          text NOT NULL,
    source_type     text NOT NULL CHECK (source_type IN ('bank','cc')),
    filename        text NOT NULL,
    storage_path    text NOT NULL,
    file_size_bytes bigint,
    uploaded_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (person, source_type, filename)
);
CREATE INDEX IF NOT EXISTS idx_csv_person ON csv_files (person, source_type);

-- Semantic similarity search RPC
CREATE OR REPLACE FUNCTION search_llm_cache(
    query_embedding      vector(384),
    query_person         text,
    similarity_threshold float DEFAULT 0.92,
    max_age_days         int   DEFAULT 7
)
RETURNS TABLE (id uuid, report_markdown text, similarity float)
LANGUAGE sql STABLE AS $$
    SELECT id, report_markdown,
           1 - (embedding <=> query_embedding) AS similarity
    FROM   llm_cache
    WHERE  person    = query_person
      AND  created_at > now() - (max_age_days || ' days')::interval
      AND  embedding IS NOT NULL
      AND  1 - (embedding <=> query_embedding) >= similarity_threshold
    ORDER  BY embedding <=> query_embedding
    LIMIT  1;
$$;

-- RLS — anon key allowed full access (private app)
ALTER TABLE merchant_overrides  ENABLE ROW LEVEL SECURITY;
ALTER TABLE saving_goals        ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_cache           ENABLE ROW LEVEL SECURITY;
ALTER TABLE csv_files           ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='merchant_overrides' AND policyname='anon_all') THEN
    CREATE POLICY anon_all ON merchant_overrides FOR ALL TO anon USING (true) WITH CHECK (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='saving_goals' AND policyname='anon_all') THEN
    CREATE POLICY anon_all ON saving_goals FOR ALL TO anon USING (true) WITH CHECK (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='llm_cache' AND policyname='anon_all') THEN
    CREATE POLICY anon_all ON llm_cache FOR ALL TO anon USING (true) WITH CHECK (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='csv_files' AND policyname='anon_all') THEN
    CREATE POLICY anon_all ON csv_files FOR ALL TO anon USING (true) WITH CHECK (true);
  END IF;
END $$;
