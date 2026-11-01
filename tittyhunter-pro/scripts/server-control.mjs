import { spawn } from 'node:child_process';
import { existsSync, openSync, readFileSync, readlinkSync, unlinkSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const pidPath = path.join(projectRoot, 'server.pid');
const logPath = path.join(projectRoot, 'server.log');
const action = process.argv[2];

function readPid() {
  if (!existsSync(pidPath)) return null;
  const pid = Number(readFileSync(pidPath, 'utf8').trim());
  return Number.isSafeInteger(pid) && pid > 1 ? pid : null;
}

function isAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function isManagedProcess(pid) {
  if (!isAlive(pid)) return false;
  try {
    const command = readFileSync(`/proc/${pid}/cmdline`, 'utf8').replaceAll('\0', ' ');
    const workingDirectory = readlinkSync(`/proc/${pid}/cwd`);
    return workingDirectory === projectRoot && command.includes('server.ts') && command.includes('--import');
  } catch {
    // Without /proc, do not risk terminating an unrelated process.
    return false;
  }
}

async function pause(ms) {
  await new Promise(resolve => setTimeout(resolve, ms));
}

if (action === 'start') {
  const existingPid = readPid();
  if (existingPid && isManagedProcess(existingPid)) {
    console.log(`Pyrmsearch server is already running (PID ${existingPid}).`);
    process.exit(0);
  }
  if (existsSync(pidPath)) unlinkSync(pidPath);

  const outputFd = openSync(logPath, 'a');
  const child = spawn(process.execPath, ['--import', 'tsx', 'server.ts'], {
    cwd: projectRoot,
    detached: true,
    env: { ...process.env, NODE_ENV: 'production' },
    stdio: ['ignore', outputFd, outputFd]
  });
  try {
    await new Promise((resolve, reject) => {
      child.once('spawn', resolve);
      child.once('error', reject);
    });
    writeFileSync(pidPath, `${child.pid}\n`, { mode: 0o600 });
    child.unref();
    console.log(`Pyrmsearch server started (PID ${child.pid}). Logs: ${path.basename(logPath)}`);
  } catch (error) {
    console.error(`Could not start server: ${error instanceof Error ? error.message : 'unknown error'}`);
    process.exitCode = 1;
  }
} else if (action === 'stop') {
  const pid = readPid();
  if (!pid) {
    if (existsSync(pidPath)) unlinkSync(pidPath);
    console.log('Pyrmsearch server is not running.');
    process.exit(0);
  }
  if (!isManagedProcess(pid)) {
    unlinkSync(pidPath);
    console.log('Stale PID file removed; no managed server process was found.');
    process.exit(0);
  }

  process.kill(pid, 'SIGTERM');
  for (let attempt = 0; attempt < 30 && isAlive(pid); attempt += 1) await pause(100);
  if (isManagedProcess(pid)) process.kill(pid, 'SIGKILL');
  if (existsSync(pidPath)) unlinkSync(pidPath);
  console.log('Pyrmsearch server stopped.');
} else if (action === 'status') {
  const pid = readPid();
  if (pid && isManagedProcess(pid)) console.log(`Running (PID ${pid}).`);
  else {
    if (existsSync(pidPath)) unlinkSync(pidPath);
    console.log('Not running.');
  }
} else {
  console.error('Usage: node scripts/server-control.mjs <start|stop|status>');
  process.exitCode = 2;
}
