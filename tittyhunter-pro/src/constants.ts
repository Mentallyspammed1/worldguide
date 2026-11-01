export interface SiteConfig {
  id: string;
  name: string;
  base: string;
  search: string;
  paginate: string;
  icon: string;
  color: string;
  cardSel: string;
  linkSel: string;
  titleSel: string;
  durationSel?: string;
  viewsSel?: string;
  buildUrl: (query: string, page: number) => string;
}

export const SITES: Record<string, SiteConfig> = {
  xvideos: {
    id: 'xvideos',
    name: 'XVideos',
    base: 'https://www.xvideos.com',
    search: '/?k=',
    paginate: '&p=',
    icon: 'XV',
    color: '#ff6600',
    cardSel:
      'div.thumb-block, div.video-item, .video-block, .video-card, .video-thumb, .frame-block',
    linkSel:
      'a[href*="/video"], a.video-link, a.video-title, .thumb-block a, .thumb a',
    titleSel:
      '.title a, p.title a, a[title], .video-title, .video-name, .title',
    durationSel: '.duration',
    viewsSel: '.sprfluous, .views',
    buildUrl: (q, p) =>
      p === 1
        ? `https://www.xvideos.com/?k=${encodeURIComponent(q)}`
        : `https://www.xvideos.com/?k=${encodeURIComponent(q)}&p=${p - 1}`
  },
  xnxx: {
    id: 'xnxx',
    name: 'XNXX',
    base: 'https://www.xnxx.com',
    search: '/search/',
    paginate: '/',
    icon: 'XN',
    color: '#ff9933',
    cardSel:
      'div.thumb-block, div.video-item, .video-block, .video-card, .video-thumb, .frame-block',
    linkSel:
      'a[href*="/video"], a.video-link, a.video-title, .thumb-block a, .thumb a',
    titleSel:
      '.title a, p a[title], a[title], .video-title, .video-name, .title',
    durationSel: '.duration',
    viewsSel: '.metadata',
    buildUrl: (q, p) =>
      p === 1
        ? `https://www.xnxx.com/search/${encodeURIComponent(q)}`
        : `https://www.xnxx.com/search/${encodeURIComponent(q)}/${p}`
  },
  eporner: {
    id: 'eporner',
    name: 'Eporner',
    base: 'https://www.eporner.com',
    search: '/search/',
    paginate: '/',
    icon: 'EP',
    color: '#d300c5',
    cardSel: 'div[id^="vf"], div.mb, .video-item, .video-card, .video-thumb',
    linkSel: 'a[href*="/video-"], a.video-link, a.video-title, .mb a',
    titleSel: 'a[title], p.mbtit a, p a, .mbunder a, .video-title',
    durationSel: '.mbtim, .duration',
    viewsSel: '.mbstats, .mbrate',
    buildUrl: (q, p) =>
      `https://www.eporner.com/search/${encodeURIComponent(q)}/${p}/`
  },
  xhamster: {
    id: 'xhamster',
    name: 'XHamster',
    base: 'https://xhamster.com',
    search: '/search/',
    paginate: '?page=',
    icon: 'XH',
    color: '#ffaa00',
    cardSel: 'div.video-thumb, [data-video-id], .thumb-list__item, .video-card',
    linkSel:
      'a[data-role="thumb-link"], a[href*="/videos/"], a.video-thumb__image-container',
    titleSel:
      'a.video-thumb-info__name, a[title], a[aria-label], .video-thumb__title',
    durationSel:
      '.duration, .video-thumb__duration, [data-role="video-duration"]',
    viewsSel: '.video-thumb-views, .views',
    buildUrl: (q, p) =>
      `https://xhamster.com/search/${encodeURIComponent(q)}?page=${p}`
  },
  pornhub: {
    id: 'pornhub',
    name: 'Pornhub',
    base: 'https://www.pornhub.com',
    search: '/video/search?search=',
    paginate: '&page=',
    icon: 'PH',
    color: '#ff3366',
    cardSel:
      'li.pcVideoListItem, div.videoBox, li.videoblock, .video-card, .video-item, .video-thumb',
    linkSel:
      'a[href*="view_video.php"], a[href*="/video"], .video-link, a.video-title, .video-thumb a',
    titleSel:
      'span.title a, a.title, a[title], .video-title, .video-name, .title',
    durationSel: '.duration, var.duration',
    viewsSel: '.views var, .views',
    buildUrl: (q, p) =>
      `https://www.pornhub.com/video/search?search=${encodeURIComponent(q)}&page=${p}`
  },
  youporn: {
    id: 'youporn',
    name: 'YouPorn',
    base: 'https://www.youporn.com',
    search: '/search/?query=',
    paginate: '&page=',
    icon: 'YP',
    color: '#05d9e8',
    cardSel:
      'div.video-box, div.thumb-block, a.video-box, .video-item, .video-card, .video-thumb',
    linkSel: 'a[href*="/watch/"], a.video-link, a.video-title, .video-box a',
    titleSel: '.video-title, a[title], .title, .video-name',
    durationSel: '.duration',
    viewsSel: '.views',
    buildUrl: (q, p) =>
      `https://www.youporn.com/search/?query=${encodeURIComponent(q)}&page=${p}`
  },
  redtube: {
    id: 'redtube',
    name: 'RedTube',
    base: 'https://www.redtube.com',
    search: '/?search=',
    paginate: '&page=',
    icon: 'RT',
    color: '#e60000',
    cardSel: 'li.videoblock, div.video-item, .video-card, .thumb-block',
    linkSel: 'a[href*="/"], a.video-link',
    titleSel: 'a[title], .video-title, .title',
    durationSel: '.duration',
    viewsSel: '.views',
    buildUrl: (q, p) =>
      `https://www.redtube.com/?search=${encodeURIComponent(q)}&page=${p}`
  },
  okxxx: {
    id: 'okxxx',
    name: 'OK.XXX',
    base: 'https://ok.xxx',
    search: '/search/',
    paginate: '/',
    icon: 'OK',
    color: '#e0245e',
    cardSel: 'div.thumb-bl-video, div.thumb-bl, .item',
    linkSel: 'a[href*="/video/"]',
    titleSel: 'a[title], a[href*="/video/"]',
    durationSel: '.video-meta li:first-child span, .duration',
    viewsSel: '.video-meta li:nth-child(3) span, .views',
    buildUrl: (q, p) =>
      p === 1
        ? `https://ok.xxx/search/${encodeURIComponent(q)}/`
        : `https://ok.xxx/search/${encodeURIComponent(q)}/${p}/`
  },
  helloporn: {
    id: 'helloporn',
    name: 'Hello.Porn',
    base: 'https://hello.porn',
    search: '/search/',
    paginate: '/',
    icon: 'HP',
    color: '#ff0055',
    cardSel: 'div.item',
    linkSel: 'a[href*="/videos/"]',
    titleSel: 'a[title]',
    durationSel: 'p.duration_item, .duration',
    viewsSel: '.views',
    buildUrl: (q, p) =>
      p === 1
        ? `https://hello.porn/search/${encodeURIComponent(q)}/`
        : `https://hello.porn/search/${encodeURIComponent(q)}/${p}/`
  },
  usersporn: {
    id: 'usersporn',
    name: 'Users.Porn',
    base: 'https://usersporn.com',
    search: '/search/',
    paginate: '/?page=',
    icon: 'UP',
    color: '#3b82f6',
    cardSel: 'div.tiled_item_bg, div[class*="tiled_item"], .tiled_body > div',
    linkSel: 'a[href*="/video/"]',
    titleSel: '.tile_descr_2l a, a[aria-label], img[alt]',
    durationSel: '.duration, .video_time',
    viewsSel: '.views',
    buildUrl: (q, p) =>
      p === 1
        ? `https://usersporn.com/search/${encodeURIComponent(q)}/`
        : `https://usersporn.com/search/${encodeURIComponent(q)}/?page=${p}`
  },
  porntrex: {
    id: 'porntrex',
    name: 'PornTrex',
    base: 'https://www.porntrex.com',
    search: '/search/',
    paginate: '/',
    icon: 'PT',
    color: '#10b981',
    cardSel: '.video-item, .video-preview-screen, .thumb-item',
    linkSel: 'a[href*="/video/"]',
    titleSel: 'a.thumb img[alt], a[title], .title',
    durationSel: '.duration',
    viewsSel: '.views',
    buildUrl: (q, p) =>
      p === 1
        ? `https://www.porntrex.com/search/${encodeURIComponent(q)}/`
        : `https://www.porntrex.com/search/${encodeURIComponent(q)}/${p}/`
  },
  hqporner: {
    id: 'hqporner',
    name: 'HQPorner',
    base: 'https://hqporner.com',
    search: '/?q=',
    paginate: '&p=',
    icon: 'HQ',
    color: '#8b5cf6',
    cardSel: 'section.box.feature, .box.feature, div[class~="6u"]',
    linkSel: 'a[href*="/hdporn/"]',
    titleSel: 'a.image.featured[title], a[href*="/hdporn/"]',
    durationSel: '.duration',
    viewsSel: '.views',
    buildUrl: (q, p) =>
      p === 1
        ? `https://hqporner.com/?q=${encodeURIComponent(q)}`
        : `https://hqporner.com/?q=${encodeURIComponent(q)}&p=${p}`
  },
  spankbang: {
    id: 'spankbang',
    name: 'SpankBang',
    base: 'https://spankbang.com',
    search: '/s/',
    paginate: '/',
    icon: 'SB',
    color: '#00ccff',
    cardSel: '.video-item, .video-card, div[data-id], .thumb-block',
    linkSel: 'a[href*="/video/"], a[href*="/watch/"], a.thumb',
    titleSel: 'a.title, a[title], .n',
    durationSel: '.l, .duration',
    viewsSel: '.v, .views',
    buildUrl: (q, p) => `https://spankbang.com/s/${encodeURIComponent(q)}/${p}/`
  }
};

export interface TrendingCategory {
  id: string;
  label: string;
  emoji: string;
  color: string;
  tags: string[];
}

export const TRENDING_CATEGORIES: TrendingCategory[] = [
  {
    id: 'all',
    label: 'All Trending',
    emoji: '🔥',
    color: '#ff2a6d',
    tags: [
      'blonde',
      'brunette',
      'redhead',
      'amateur',
      'milf',
      'latina',
      'asian',
      'college',
      'vr',
      '4k',
      'ebony',
      'public',
      'outdoor',
      'threesome',
      'anal',
      'creampie',
      'squirt',
      'big tits',
      'petite',
      'stepmom',
      'lesbian',
      'gangbang',
      'interracial',
      'hentai'
    ]
  },
  {
    id: 'featured',
    label: 'Featured',
    emoji: '⭐',
    color: '#d300c5',
    tags: ['adult amateur', 'milf', 'blonde', 'big tits', 'creampie']
  },
  {
    id: 'ethnic',
    label: 'Ethnic',
    emoji: '🌍',
    color: '#05d9e8',
    tags: [
      'ebony',
      'latina',
      'asian',
      'japanese',
      'interracial',
      'indian',
      'arab',
      'russian'
    ]
  },
  {
    id: 'body',
    label: 'Body Type',
    emoji: '💃',
    color: '#ff6600',
    tags: [
      'petite',
      'bbw',
      'curvy',
      'athletic',
      'skinny',
      'muscular',
      'tall',
      'short'
    ]
  },
  {
    id: 'act',
    label: 'Acts',
    emoji: '🔞',
    color: '#ffaa00',
    tags: [
      'anal',
      'creampie',
      'squirt',
      'threesome',
      'gangbang',
      'deepthroat',
      'facial',
      'swallow'
    ]
  },
  {
    id: 'setting',
    label: 'Setting',
    emoji: '📍',
    color: '#10b981',
    tags: [
      'public',
      'outdoor',
      'public agent',
      'fake taxi',
      'massage',
      'office',
      'gym',
      'beach'
    ]
  },
  {
    id: 'quality',
    label: 'Quality',
    emoji: '🎬',
    color: '#8b5cf6',
    tags: ['4k', 'hd', 'vr', '3d', '60fps', 'ultra hd', '1080p', '720p']
  },
  {
    id: 'roleplay',
    label: 'Roleplay',
    emoji: '🎭',
    color: '#3b82f6',
    tags: [
      'stepmom',
      'stepsister',
      'stepdad',
      'teacher',
      'nurse',
      'maid',
      'police',
      'secretary'
    ]
  }
];

export const POPULAR_TAGS: string[] =
  TRENDING_CATEGORIES.find(c => c.id === 'all')?.tags ?? [];

export const RANDOM_TAGS = [
  'surprise me',
  'random',
  'mystery',
  'wildcard',
  'anything goes'
];
