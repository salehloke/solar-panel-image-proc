-- PostgreSQL initialization script for SolarAI
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'moderator')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create solar_panels table
CREATE TABLE IF NOT EXISTS solar_panels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    address TEXT,
    capacity DECIMAL(10, 2), -- in kW
    installation_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create analysis_results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    prediction VARCHAR(10) NOT NULL CHECK (prediction IN ('clean', 'dirty')),
    confidence DECIMAL(3, 2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version VARCHAR(50),
    processing_time DECIMAL(10, 3), -- in seconds
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_solar_panels_user_id ON solar_panels(user_id);
CREATE INDEX IF NOT EXISTS idx_solar_panels_location ON solar_panels(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_analysis_results_user_id ON analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_prediction ON analysis_results(prediction);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_solar_panels_updated_at 
    BEFORE UPDATE ON solar_panels 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password will be hashed in the application)
INSERT INTO users (email, username, password_hash, role) 
VALUES (
    'admin@solarai.com',
    'admin',
    'temporary_hash_will_be_updated_by_app',
    'admin'
) ON CONFLICT (email) DO NOTHING;

-- Insert sample solar panel data
INSERT INTO solar_panels (user_id, name, latitude, longitude, address, capacity, installation_date)
SELECT 
    u.id,
    'Sample Solar Panel',
    37.7749,
    -122.4194,
    'San Francisco, CA',
    5.0,
    '2023-01-01'
FROM users u 
WHERE u.email = 'admin@solarai.com'
ON CONFLICT DO NOTHING;

-- Create a view for analysis statistics
CREATE OR REPLACE VIEW analysis_stats AS
SELECT 
    COUNT(*) as total_analyses,
    COUNT(CASE WHEN prediction = 'clean' THEN 1 END) as clean_count,
    COUNT(CASE WHEN prediction = 'dirty' THEN 1 END) as dirty_count,
    AVG(CASE WHEN prediction = 'clean' THEN confidence END) as avg_confidence_clean,
    AVG(CASE WHEN prediction = 'dirty' THEN confidence END) as avg_confidence_dirty
FROM analysis_results;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO solarai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO solarai;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO solarai;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL initialization completed successfully!';
    RAISE NOTICE 'Tables created: users, solar_panels, analysis_results';
    RAISE NOTICE 'Indexes created for better performance';
    RAISE NOTICE 'Default admin user created: admin@solarai.com';
END $$; 