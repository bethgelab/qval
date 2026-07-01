def signal_function(state: str, action: str, next_state: str) -> float:
    q_value = 0.5
    success_keywords = ['success', 'completed', 'done', 'created', 'wrote', 'saved', 'ok', 'true', 'passed', 'verified', 'exit code', '0', 'root@', 'bash@']
    error_keywords = ['error', 'failed', 'invalid', 'cannot', 'permission', 'syntax', 'segmentation', 'killed', 'failed', 'denied', 'refused']
    progress_commands = ['mkdir', 'touch', 'cp', 'mv', 'rm', 'chmod', 'chown', 'grep', 'find', 'sed', 'awk', 'python', 'pip', 'apt', 'npm', 'git', 'ssh', 'curl', 'wget', 'tar', 'gzip', 'vim', 'nano', 'cat', 'less', 'head', 'tail', 'wc', 'sort', 'uniq', 'cut', 'paste', 'join', 'diff', 'cmp', 'md5sum', 'sha1sum', 'sha256sum', 'openssl', 'nc', 'ping', 'traceroute', 'nslookup', 'dig', 'host', 'ifconfig', 'ip', 'route', 'netstat', 'ss', 'lsof', 'top', 'htop', 'ps', 'kill', 'pkill', 'killall', 'nohup', 'screen', 'tmux', 'scp', 'rsync', 'ssh', 'git', 'cat', 'echo', 'printf', 'tee', 'dd', 'base64', 'xxd', 'hexdump', 'od', 'strings', 'file', 'uname', 'whoami', 'pwd', 'ls', 'cd', 'clear', 'history', 'jobs', 'fg', 'bg', 'exit', 'logout', 'sudo', 'su', 'passwd', 'useradd', 'userdel', 'usermod', 'groupadd', 'groupdel', 'groupmod', 'passwd', 'shadow', 'crontab', 'at', 'systemctl', 'service', 'init', 'shutdown', 'reboot', 'halt', 'poweroff', 'mount', 'umount', 'df', 'du', 'free', 'vmstat', 'iostat', 'sar', 'netstat', 'ss', 'tcpdump', 'wireshark', 'nmap', 'tcpdump', 'netstat', 'ip', 'ipconfig', 'ifconfig', 'route', 'netstat', 'ss', 'lsof', 'fuser', 'lsof', 'ps', 'top', 'htop', 'pidof', 'pgrep', 'pkill', 'kill', 'killall', 'nice', 'renice', 'ionice', 'chrt', 'setarch', 'ulimit', 'limit', 'env', 'export', 'export', 'source', 'bash', 'sh', 'zsh', 'fish', 'dash', 'ash', 'ksh', 'tcsh', 'csh', 'xonsh', 'python', 'python2', 'python3', 'pythonw', 'ipython', 'jupyter', 'notebook', 'pypy', 'pypy3', 'perl', 'ruby', 'php', 'node', 'nodejs', 'deno', 'go', 'rust', 'java', 'scala', 'kotlin', 'groovy', 'groovy', 'groovy', 'groovy']
    for keyword in success_keywords:
        if keyword in next_state.lower():
            q_value += 0.15
    for keyword in error_keywords:
        if keyword in next_state.lower():
            q_value -= 0.25
    for cmd in progress_commands:
        if action.lower().startswith(cmd):
            q_value += 0.1
    if next_state != state:
        q_value += 0.05
    q_value = max(0.0, min(1.0, q_value))
    return q_value