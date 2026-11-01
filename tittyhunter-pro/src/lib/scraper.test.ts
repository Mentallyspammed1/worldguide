import { describe, expect, it } from 'vitest';
import { SITES } from '../constants';
import { abs, parseHtml } from './scraper';

describe('scraper URL handling', () => {
  it('resolves relative links and upgrades HTTP to HTTPS', () => {
    expect(abs('https://www.xvideos.com', '/video123')).toBe('https://www.xvideos.com/video123');
    expect(abs('https://www.xvideos.com', 'http://cdn.xvideos.com/thumb.jpg')).toBe(
      'https://cdn.xvideos.com/thumb.jpg'
    );
  });

  it.each([
    'javascript:alert(1)',
    'data:text/html,bad',
    'file:///etc/passwd',
    '//localhost/a',
    'https://127.0.0.1/a',
    'https://[::1]/a',
    'https://2130706433/a',
    'https://media.intranet/a',
    'https://media.local/a'
  ])('rejects unsafe URL %s', value => {
    expect(abs('https://www.xvideos.com', value)).toBe('');
  });
});

describe('parseHtml', () => {
  it('extracts stable, de-duplicated, same-site results and skips unsafe links', () => {
    const html = `
      <div class="thumb-block">
        <a class="video-link" href="/video123"><span class="title">  Safe   title </span></a>
        <img data-src="/thumb.jpg" alt="poster">
        <video data-preview-video="https://cdn.xvideos.com/preview.mp4?token=one"></video>
      </div>
      <div class="thumb-block">
        <a class="video-link" href="/video123"><span class="title">Duplicate</span></a>
      </div>
      <div class="thumb-block">
        <a class="video-link" href="https://not-xvideos.example/video456">External</a>
      </div>
      <script>window.__unsafe = true</script>
    `;

    const results = parseHtml(html, 'xvideos', SITES.xvideos.base);
    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({
      title: 'Safe title',
      url: 'https://www.xvideos.com/video123',
      thumb: 'https://www.xvideos.com/thumb.jpg',
      previewVideo: 'https://cdn.xvideos.com/preview.mp4?token=one',
      site: 'xvideos',
      siteName: 'XVideos'
    });
    expect(parseHtml(html, 'xvideos', SITES.xvideos.base)[0].id).toBe(results[0].id);
  });

  it('does not mistake ordinary links for preview media', () => {
    const html = `
      <div class="thumb-block">
        <a class="video-link" href="/video123">Title</a>
        <div data-preview="https://www.xvideos.com/video123">not a media file</div>
      </div>`;
    expect(parseHtml(html, 'xvideos', SITES.xvideos.base)[0].previewVideo).toBe('');
  });

  it('handles malformed selectors without crashing', () => {
    const html = '<div class="6u"><a href="/hdporn/item-1">Sample</a></div>';
    expect(() => parseHtml(html, 'hqporner', SITES.hqporner.base)).not.toThrow();
  });
});
