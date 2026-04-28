/**
 * FarmBrain Logger
 * Simple structured console logger.
 */
const levels = { INFO: '✅', WARN: '⚠️ ', ERROR: '❌', DEBUG: '🔍' };

function log(level, module, message, data = null) {
  const ts = new Date().toISOString();
  const icon = levels[level] || '•';
  const line = `${ts} ${icon} [${level}] [${module}] ${message}`;
  console.log(line);
  if (data) console.log('   →', JSON.stringify(data, null, 2));
}

module.exports = {
  info:  (mod, msg, d) => log('INFO',  mod, msg, d),
  warn:  (mod, msg, d) => log('WARN',  mod, msg, d),
  error: (mod, msg, d) => log('ERROR', mod, msg, d),
  debug: (mod, msg, d) => log('DEBUG', mod, msg, d),
};
