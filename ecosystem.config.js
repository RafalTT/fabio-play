module.exports = {
  apps: [
    {
      name: 'fabio-play-api',
      script: '.venv/bin/uvicorn',
      args: 'app.main:app --host 0.0.0.0 --port 8000',
      cwd: '/home/rafalt/fabio-play/backend',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1',
      },
      restart_delay: 3000,
      max_restarts: 10,
      watch: false,
      error_file: '/home/rafalt/.pm2/logs/fabio-api-error.log',
      out_file: '/home/rafalt/.pm2/logs/fabio-api-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
    },
  ],
};
