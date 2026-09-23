import os

bind = "0.0.0.0:" + os.environ.get("PORT", "8080")
workers = 1
threads = 4
timeout = 120
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
control_socket_disable = True
