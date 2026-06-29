-- ============================================================
-- MySQL 初始化 — medical_rag 数据库
-- Docker 首次启动时自动执行
-- ============================================================

-- ai_model_config 表 (AI 场景配置)
CREATE TABLE IF NOT EXISTS ai_model_config (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    scene         VARCHAR(30)  NOT NULL COMMENT '业务场景标识',
    model_name    VARCHAR(100) NOT NULL DEFAULT 'qwen-flash',
    api_base_url  VARCHAR(255) NOT NULL DEFAULT 'https://api.deepseek.com',
    api_key       VARCHAR(255) NOT NULL DEFAULT '',
    temperature   DOUBLE       NOT NULL DEFAULT 0.3,
    max_tokens    INT          NOT NULL DEFAULT 800,
    top_p         DOUBLE       NOT NULL DEFAULT 0.9,
    system_prompt TEXT         NOT NULL COMMENT 'System Prompt 模板',
    status        TINYINT      NOT NULL DEFAULT 1 COMMENT '1启用 0禁用',
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME     DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_scene (scene)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='AI模型配置表';

-- dialogue_session 表 (多轮对话会话)
CREATE TABLE IF NOT EXISTS dialogue_session (
    session_id           VARCHAR(36)  PRIMARY KEY COMMENT 'UUID v4',
    patient_id           BIGINT       DEFAULT NULL,
    status               VARCHAR(20)  NOT NULL DEFAULT 'active'
                         COMMENT 'active/closed/emergency/timeout',
    collected_symptoms   TEXT         COMMENT 'JSON: accumulated symptoms',
    extracted_keywords   TEXT         COMMENT 'JSON: keyword list',
    candidate_diseases   TEXT         COMMENT 'JSON: top-5 candidate diseases',
    dialogue_history     TEXT         COMMENT 'JSON: full Q&A history',
    final_recommendation TEXT         COMMENT 'JSON: final recommendation',
    max_turns            INT          DEFAULT 8,
    current_turn         INT          DEFAULT 0,
    created_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                         ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_patient_id (patient_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='多轮对话会话状态表';

-- health_suggestion 表 (个性化生活建议)
CREATE TABLE IF NOT EXISTS health_suggestion (
    suggestion_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id     BIGINT      NOT NULL COMMENT '关联健康档案ID',
    patient_id    BIGINT      NOT NULL COMMENT '患者ID (冗余)',
    category      VARCHAR(20) NOT NULL COMMENT 'diet/exercise/sleep/medication/seasonal',
    title         VARCHAR(100)NOT NULL COMMENT '建议标题',
    content       TEXT        NOT NULL COMMENT '建议正文',
    is_active     TINYINT     NOT NULL DEFAULT 1,
    generated_at  DATETIME    NOT NULL COMMENT 'AI生成时间',
    expires_at    DATETIME    DEFAULT NULL,
    INDEX idx_record_id  (record_id),
    INDEX idx_patient_id (patient_id),
    INDEX idx_category   (category),
    INDEX idx_active     (is_active, generated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='个性化生活建议表';
