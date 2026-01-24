/**
 * PM2 Ecosystem Configuration
 *
 * Usage:
 *   pm2 start ecosystem.config.js
 *   pm2 start ecosystem.config.js --env production
 *   pm2 reload hana-bridge
 *   pm2 scale hana-bridge 4
 */

module.exports = {
    apps: [
        {
            name: 'hana-bridge',
            script: 'server.prod.js',

            // Cluster mode - fork multiple workers
            instances: 'max',        // Use all CPU cores
            exec_mode: 'cluster',    // Enable cluster mode

            // Watch & Reload
            watch: false,
            ignore_watch: ['node_modules', 'logs'],

            // Restart behavior
            max_restarts: 10,
            min_uptime: '10s',
            max_memory_restart: '500M',

            // Logging
            log_file: './logs/combined.log',
            out_file: './logs/out.log',
            error_file: './logs/error.log',
            log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
            merge_logs: true,

            // Graceful shutdown
            kill_timeout: 5000,
            listen_timeout: 3000,
            shutdown_with_message: true,
            wait_ready: true,

            // Environment
            env: {
                NODE_ENV: 'development',
                BRIDGE_PORT: 3001,
                LOG_LEVEL: 'DEBUG',
                POOL_MIN: 1,
                POOL_MAX: 5,
            },

            env_production: {
                NODE_ENV: 'production',
                BRIDGE_PORT: 3001,
                LOG_LEVEL: 'INFO',
                POOL_MIN: 2,
                POOL_MAX: 10,
                RATE_LIMIT_MAX: 100,
            },

            env_staging: {
                NODE_ENV: 'staging',
                BRIDGE_PORT: 3001,
                LOG_LEVEL: 'DEBUG',
                POOL_MIN: 1,
                POOL_MAX: 5,
            },
        },

        // Single instance mode (for debugging)
        {
            name: 'hana-bridge-single',
            script: 'server.prod.js',
            instances: 1,
            exec_mode: 'fork',
            watch: true,
            ignore_watch: ['node_modules', 'logs'],
            env: {
                NODE_ENV: 'development',
                BRIDGE_PORT: 3001,
                LOG_LEVEL: 'DEBUG',
            },
        },
    ],

    // Deployment configuration
    deploy: {
        production: {
            user: 'deploy',
            host: ['server1.example.com', 'server2.example.com'],
            ref: 'origin/main',
            repo: 'git@github.com:user/hana-bridge.git',
            path: '/var/www/hana-bridge',
            'pre-deploy': 'git fetch --all',
            'post-deploy': 'npm ci && pm2 reload ecosystem.config.js --env production',
            'pre-setup': 'mkdir -p /var/www/hana-bridge/logs',
            env: {
                NODE_ENV: 'production',
            },
        },

        staging: {
            user: 'deploy',
            host: 'staging.example.com',
            ref: 'origin/develop',
            repo: 'git@github.com:user/hana-bridge.git',
            path: '/var/www/hana-bridge-staging',
            'post-deploy': 'npm ci && pm2 reload ecosystem.config.js --env staging',
            env: {
                NODE_ENV: 'staging',
            },
        },
    },
};
