# Jupyter Notebook Configuration

c = get_config()

# Disable authentication
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.disable_check_xsrf = True

# Allow remote access
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '0.0.0.0'

# Disable browser opening
c.NotebookApp.open_browser = True

# Set port
c.NotebookApp.port = 8888

# Optional: Increase upload size limit
c.NotebookApp.max_body_size = 1024 * 1024 * 1024  # 1GB max upload size