import re
import math

def signal_function(state: str) -> float:
    value = 0.5
    
    error_patterns = ['error', 'failed', 'invalid', 'cannot', 'unavailable', 'blocked', 'alert', 'warning']
    if any(pattern in state.lower() for pattern in error_patterns):
        value -= 0.3
    
    success_patterns = ['completed', 'success', 'saved', 'added', 'created', 'sent', 'done', 'finish', 'submit']
    if any(pattern in state.lower() for pattern in success_patterns):
        value += 0.3
    
    element_count = len(re.findall(r'\bid="[^"]*"', state))
    if element_count > 15:
        value += 0.15
    elif element_count > 8:
        value += 0.08
    elif element_count > 3:
        value += 0.03
    
    form_fields = len(re.findall(r'<input|<textarea|<select', state))
    if form_fields > 3:
        value -= 0.05
    elif form_fields == 0:
        value += 0.02
    
    nav_elements = len(re.findall(r'<a href|<button|<nav|<ul|<ol', state))
    if nav_elements > 5:
        value += 0.05
    
    goal_keywords = ['calendar', 'event', 'message', 'todo', 'map', 'code', 'editor', 'send', 'add', 'create', 'edit', 'delete', 'update', 'view', 'search', 'filter', 'sort', 'login', 'sign', 'profile', 'settings', 'home', 'dashboard', 'list', 'table', 'grid', 'form', 'modal', 'dialog', 'notification', 'status', 'result', 'output', 'input', 'field', 'text', 'button', 'link', 'menu', 'toolbar', 'sidebar', 'footer', 'header', 'title', 'subtitle', 'label', 'placeholder', 'required', 'optional', 'checkbox', 'radio', 'dropdown', 'autocomplete', 'validation', 'confirmation', 'cancel', 'confirm', 'ok', 'yes', 'no', 'clear', 'reset', 'save', 'apply', 'edit', 'delete', 'new', 'open', 'close', 'back', 'next', 'previous', 'forward', 'refresh', 'reload', 'print', 'export', 'import', 'share', 'copy', 'paste', 'cut', 'select', 'highlight', 'focus', 'blur', 'click', 'double', 'right', 'middle', 'wheel', 'scroll', 'resize', 'drag', 'drop', 'upload', 'download', 'link', 'bookmark', 'history', 'tab', 'window', 'frame', 'iframe', 'canvas', 'svg', 'image', 'video', 'audio', 'font', 'style', 'class', 'id', 'name', 'role', 'type', 'value', 'checked', 'disabled', 'readonly', 'hidden', 'visible', 'active', 'selected', 'expanded', 'collapsed', 'open', 'closed', 'loading', 'ready', 'pending', 'processing', 'complete', 'success', 'error', 'warning', 'info', 'debug', 'log', 'trace', 'console', 'browser', 'network', 'request', 'response', 'header', 'body', 'url', 'path', 'domain', 'host', 'port', 'protocol', 'scheme', 'method', 'status', 'code', 'message', 'data', 'content', 'type', 'format', 'mime', 'charset', 'encoding', 'language', 'locale', 'timezone', 'date', 'time', 'datetime', 'duration', 'interval', 'frequency', 'period', 'cycle', 'repeat', 'loop', 'iteration', 'step', 'stage', 'phase', 'version', 'revision', 'build', 'release', 'patch', 'minor', 'major', 'hotfix', 'feature', 'bug', 'issue', 'ticket', 'task', 'work', 'item', 'object', 'entity', 'record', 'entry', 'row', 'column', 'cell', 'grid', 'list', 'array', 'set', 'map', 'dict', 'list', 'tuple', 'dict', 'json', 'xml', 'html', 'css', 'js', 'py', 'java', 'c', 'cpp', 'go', 'rust', 'ruby', 'php', 'sql', 'yaml', 'toml', 'ini', 'env', 'conf', 'cfg', 'log', 'txt', 'md', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'zip', 'rar', 'tar', 'gz', 'bz2', '7z', 'iso', 'img', 'exe', 'dll', 'so', 'lib', 'bin', 'app', 'apk', 'ipa', 'deb', 'rpm', 'pkg', 'snap', 'flatpak', 'docker', 'k8s', 'aws', 'azure', 'gcp', 'vpc', 'sub', 'net', 'org', 'com', 'io', 'co', 'net', 'org', 'com', 'io', 'co', 'uk', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'cn', 'jp', 'kr', 'in', 'br', 'mx', 'ar', 'cl', 'co', 'pe', 've', 'ec', 'uy', 'py', 'bo', 'gt', 'hn', 'sv', 'ni', 'cr', 'pa', 'do', 'pr', 'cu', 'jm', 'tt', 'bb', 'gd', 'vc', 'lc', 'vc', 'ag', 'dm', 'kp', 'ky', 'ai', 'ms', 'vg', 'vg', 'vi', 'mp', 'fm', 'pw', 'mh', 'gu', 'pr', 'vi', 'mp', 'fm', 'pw', 'mh', 'gu', 'pr', 'vi', 'mp', 'fm', 'pw', 'mh']
    if any(keyword in state.lower() for keyword in goal_keywords):
        value += 0.05
    
    progress_keywords = ['step', 'progress', 'percent', 'percentage', 'complete', 'done', 'finish', 'end', 'final', 'last', 'current', 'next', 'previous', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']
    if any(keyword in state.lower() for keyword in progress_keywords):
        value += 0.03
    
    value = max(0.0, min(1.0, value))
    return value