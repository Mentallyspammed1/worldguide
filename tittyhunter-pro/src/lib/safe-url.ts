const BLOCKED_HOST_SUFFIXES = [
  '.localhost',
  '.local',
  '.localdomain',
  '.internal',
  '.intranet',
  '.test',
  '.example',
  '.invalid',
  '.lan',
  '.home',
  '.home.arpa',
  '.onion'
];

export function isSafePublicHostname(value: string): boolean {
  const hostname = value.toLowerCase().replace(/\.$/, '');
  return Boolean(
    hostname.includes('.') &&
      !/^\[.*\]$/.test(hostname) &&
      !/^\d{1,3}(?:\.\d{1,3}){3}$/.test(hostname) &&
      !BLOCKED_HOST_SUFFIXES.some(suffix => hostname.endsWith(suffix))
  );
}

export function isSafeHttpsUrl(value: unknown): value is string {
  if (typeof value !== 'string' || value.length > 2_048) return false;
  try {
    const url = new URL(value);
    return (
      url.protocol === 'https:' &&
      !url.username &&
      !url.password &&
      !url.port &&
      isSafePublicHostname(url.hostname)
    );
  } catch {
    return false;
  }
}
