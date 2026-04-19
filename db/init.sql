-- Temporal creates its own databases via auto-setup. We create ours here.
CREATE DATABASE agentdb;

\c agentdb

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Core task graph. parent_task_id gives us the tree; recursive CTEs give us ancestry lookups.
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL,
    parent_task_id UUID REFERENCES tasks(id),
    role TEXT NOT NULL,                    -- 'ba' | 'architect' | 'se' | 'reviewer' | 'researcher' | 'documenter'
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',-- pending | running | done | failed
    input JSONB,
    output JSONB,
    provider TEXT,
    model_used TEXT,
    tokens_in INT,
    tokens_out INT,
    cost_usd NUMERIC(10, 6),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_tasks_project ON tasks(project_id);
CREATE INDEX idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX idx_tasks_status ON tasks(status);

-- Vector store for retrieval. Start with OpenAI/Voyage 1536-dim; swap later if you change models.
CREATE TABLE task_embeddings (
    task_id UUID PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_task_embeddings_vec ON task_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- Projects table for the top-level container
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    brief TEXT NOT NULL,
    workflow_id TEXT,                      -- Temporal workflow ID, for status polling
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ DEFAULT now()
);
