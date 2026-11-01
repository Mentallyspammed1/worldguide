/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo
} from 'react';
import {
  Search,
  Copy,
  Check,
  ExternalLink,
  Trash2,
  Activity,
  RotateCcw,
  Heart,
  ChevronDown,
  ChevronUp,
  Clock,
  Eye,
  Sparkles,
  Flame,
  CheckCircle2,
  AlertCircle,
  Play,
  Film,
  X,
  Volume2,
  VolumeX,
  Download,
  Smartphone,
  ShieldCheck,
  Camera,
  PictureInPicture2,
  Share2,
  Upload,
  Save as SaveIcon,
  History
} from 'lucide-react';
import {
  SITES,
  POPULAR_TAGS,
  TRENDING_CATEGORIES,
  RANDOM_TAGS,
  SiteConfig,
  TrendingCategory
} from './constants';
import {
  scrapeSite,
  testAllSites,
  fetchFullVideoUrl,
  VideoResult,
  SiteTestResult
} from './lib/scraper';
import { usePWAInstall } from './usePWAInstall';
import {
  parseBackupText,
  sanitizeCollection,
  sanitizeHistory,
  sanitizeRecentlyWatched,
  sanitizeScreenshot,
  sanitizeVideoResult,
  type Collection,
  type RecentlyWatchedItem,
  type SavedScreenshot
} from './lib/backup';

interface VideoCardProps {
  item: VideoResult;
  isFavorited: boolean;
  onToggleFavorite: (e: React.MouseEvent, item: VideoResult) => void;
  onCopyLink: (e: React.MouseEvent, item: VideoResult) => Promise<boolean>;
  onOpenTeaser: (item: VideoResult) => void;
  onMarkWatched: (item: VideoResult) => void;
  onSaveScreenshot: (item: VideoResult, video: HTMLVideoElement | null) => void;
  onShare: (item: VideoResult) => void;
  collections: Collection[];
  onAddToCollection: (collectionId: string, item: VideoResult) => void;
}

// Self-contained Video Card component to isolate hover, preview, image error, and copy states
const VideoCard: React.FC<VideoCardProps> = ({
  item,
  isFavorited,
  onToggleFavorite,
  onCopyLink,
  onOpenTeaser,
  onMarkWatched,
  onSaveScreenshot,
  onShare,
  collections,
  onAddToCollection
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isTouchActive, setIsTouchActive] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoSrc, setVideoSrc] = useState('');
  const [proxyAttempted, setProxyAttempted] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [isPipActive, setIsPipActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const preloadTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef(true);
  const videoSrcRef = useRef('');
  const videoErrorRef = useRef(false);

  const hasVideo = Boolean(videoSrc) && !videoError;
  const shouldShowVideo = (isHovered || isTouchActive) && hasVideo;

  useEffect(() => {
    videoSrcRef.current = videoSrc;
  }, [videoSrc]);
  useEffect(() => {
    videoErrorRef.current = videoError;
  }, [videoError]);

  // Track mounted state and stop playback when a card leaves the results grid.
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (preloadTimeoutRef.current) clearTimeout(preloadTimeoutRef.current);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.removeAttribute('src');
        videoRef.current.load();
      }
    };
  }, []);

  // Reset media and hover states if React reuses this card for another result.
  useEffect(() => {
    setIsHovered(false);
    setIsTouchActive(false);
    setImgError(false);
    setVideoError(false);
    videoErrorRef.current = false;
    setVideoLoaded(false);
    setVideoLoading(false);
    setProxyAttempted(false);
    setVideoSrc('');
    videoSrcRef.current = '';
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.removeAttribute('src');
      videoRef.current.load();
    }
  }, [item.id, item.previewVideo]);

  // Defer loading preview streams until a card is near the viewport.
  useEffect(() => {
    if (!item.previewVideo) return;
    if (typeof IntersectionObserver === 'undefined') {
      videoSrcRef.current = item.previewVideo;
      setVideoSrc(item.previewVideo);
      setVideoLoading(true);
      return;
    }

    const observer = new IntersectionObserver(
      entries => {
        const entry = entries[0];
        if (!entry?.isIntersecting || videoSrcRef.current || videoErrorRef.current) return;
        observer.unobserve(entry.target);
        preloadTimeoutRef.current = setTimeout(() => {
          if (!isMountedRef.current || videoSrcRef.current || videoErrorRef.current) return;
          videoSrcRef.current = item.previewVideo;
          setVideoSrc(item.previewVideo);
          setVideoLoading(true);
        }, 150);
      },
      { threshold: 0.1, rootMargin: '100px' }
    );

    const element = containerRef.current;
    if (element) observer.observe(element);
    return () => {
      observer.disconnect();
      if (preloadTimeoutRef.current) clearTimeout(preloadTimeoutRef.current);
    };
  }, [item.previewVideo, item.id]);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const copied = await onCopyLink(e, item);
    setIsCopied(copied);
    if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    if (copied) {
      copyTimeoutRef.current = setTimeout(() => setIsCopied(false), 2000);
    }
  };

  const handleVideoLoadedData = () => {
    setVideoLoaded(true);
    setVideoLoading(false);
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (shouldShowVideo && videoLoaded) {
      video.play().catch(() => undefined);
    } else if (!shouldShowVideo) {
      video.pause();
    }
  }, [shouldShowVideo, videoLoaded, videoSrc]);

  const handleVideoError = () => {
    setVideoLoading(false);
    if (!proxyAttempted && videoSrc && !videoSrc.startsWith('/api/proxy-media')) {
      setProxyAttempted(true);
      setVideoLoaded(false);
      setVideoLoading(true);
      const proxiedUrl = `/api/proxy-media?url=${encodeURIComponent(videoSrc)}`;
      videoSrcRef.current = proxiedUrl;
      setVideoSrc(proxiedUrl);
      return;
    }
    videoErrorRef.current = true;
    setVideoError(true);
    setVideoLoaded(false);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    e.stopPropagation();
    setIsTouchActive(true);
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    e.stopPropagation();
    setIsTouchActive(false);
  };

  return (
    <div
      ref={containerRef}
      data-card-id={item.id}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      className='group bg-[#0a0a12] border border-[#1a1a2e] hover:border-[#ff2a6d] rounded-xl overflow-hidden relative transition-all duration-300 hover:-translate-y-1.5 hover:shadow-[0_8px_25px_rgba(255,42,109,0.3)] flex flex-col'
    >
      {/* Thumbnail / Video Preview Container */}
      <div className='relative w-full h-42 bg-[#070710] overflow-hidden'>
        {/* Base Poster Thumbnail Image (Always present underneath to prevent black flash) */}
        {item.thumb && !imgError ? (
          <img
            src={item.thumb}
            alt={item.title}
            referrerPolicy='no-referrer'
            loading='lazy'
            onError={() => setImgError(true)}
            className='w-full h-full object-cover transition-transform duration-500 group-hover:scale-105'
          />
        ) : (
          <div className='w-full h-full flex flex-col items-center justify-center text-xs text-[#6a6a8a] gap-1.5 bg-[#0a0a16]'>
            <Film size={24} className='text-[#1e1e38]' />
            <span className='font-mono text-[10px] text-[#555577]'>
              {item.siteName}
            </span>
          </div>
        )}

        {/* Active Hover / Touch Video Preview (Fades in over thumbnail once buffered) */}
        {hasVideo && (
          <video
            ref={videoRef}
            src={videoSrc}
            autoPlay={shouldShowVideo}
            muted
            playsInline
            loop
            preload='metadata'
            referrerPolicy='no-referrer'
            onLoadedData={handleVideoLoadedData}
            onCanPlay={handleVideoLoadedData}
            onError={handleVideoError}
            className={`absolute inset-0 w-full h-full object-cover z-10 pointer-events-none transition-opacity duration-300 ${
              shouldShowVideo && videoLoaded ? 'opacity-100' : 'opacity-0'
            }`}
          />
        )}

        {/* Hover preview status badge */}
        {shouldShowVideo && (
          <div className='absolute top-2.5 left-2.5 px-2 py-0.5 rounded text-[10px] font-extrabold uppercase tracking-widest bg-black/90 text-[#05d9e8] border border-[#05d9e8] shadow-[0_0_10px_rgba(5,217,232,0.8)] flex items-center gap-1.5 z-20 animate-pulse'>
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                videoLoaded
                  ? 'bg-[#05d9e8]'
                  : videoLoading
                    ? 'bg-[#ff2a6d] animate-ping'
                    : 'bg-[#a0a0cf]'
              }`}
            />
            {videoLoaded
              ? 'HOVER PREVIEW'
              : videoLoading
                ? 'PRELOADING...'
                : 'BUFFERING...'}
          </div>
        )}

        {/* Video availability indicator badge if not currently hovering */}
        {!shouldShowVideo && hasVideo && (
          <div className='absolute top-2.5 left-2.5 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-black/80 text-[#05d9e8] border border-[#05d9e8]/50 flex items-center gap-1 z-10 backdrop-blur-xs'>
            <Play size={9} className='fill-[#05d9e8]' />
            PREVIEW
          </div>
        )}

        {/* Picture-in-Picture Button on Top-Right (only when video is active) */}
        {shouldShowVideo &&
          videoRef.current &&
          'pictureInPictureEnabled' in document &&
          document.pictureInPictureEnabled &&
          'requestPictureInPicture' in videoRef.current && (
            <button
              type='button'
              aria-label={isPipActive ? 'Exit Picture-in-Picture' : 'Picture-in-Picture'}
              onClick={async event => {
                event.stopPropagation();
                try {
                  if (document.pictureInPictureElement) {
                    await document.exitPictureInPicture();
                    setIsPipActive(false);
                  } else if (videoRef.current) {
                    await videoRef.current.requestPictureInPicture();
                    setIsPipActive(true);
                  }
                } catch {
                  setIsPipActive(false);
                }
              }}
              title={isPipActive ? 'Exit Picture-in-Picture' : 'Picture-in-Picture'}
              className='absolute top-2.5 right-11 w-8 h-8 rounded-full border border-[#05d9e8]/50 flex items-center justify-center text-xs transition-all z-20 cursor-pointer hover:bg-[#05d9e8]/20 hover:text-[#05d9e8] hover:border-[#05d9e8]'
            >
              <PictureInPicture2 size={14} />
            </button>
          )}

        {/* Favorite Toggle Button on Top-Right */}
        <button
          type='button'
          aria-label={isFavorited ? 'Remove from favorites' : 'Save to favorites'}
          onClick={e => onToggleFavorite(e, item)}
          title={isFavorited ? 'Remove from favorites' : 'Save to favorites'}
          className={`absolute top-2.5 right-2.5 w-8 h-8 rounded-full border border-black/40 flex items-center justify-center text-sm transition-all z-20 cursor-pointer ${isFavorited ? 'bg-[#ff2a6d] text-white shadow-[0_0_12px_#ff2a6d] animate-pulse-neon' : 'bg-black/75 text-white/80 hover:bg-[#ff2a6d] hover:text-white hover:scale-110'}`}
        >
          {isFavorited ? '🔞🔞' : '🔞'}
        </button>
        <button
          type='button'
          aria-label='Save video screenshot'
          onClick={event => {
            event.stopPropagation();
            onSaveScreenshot(item, videoRef.current);
          }}
          title='Save video screenshot'
          className='absolute top-2.5 left-2.5 w-8 h-8 rounded-full border border-[#05d9e8]/50 flex items-center justify-center text-xs transition-all z-20 cursor-pointer hover:bg-[#05d9e8]/20 hover:text-[#05d9e8] hover:border-[#05d9e8]'
        >
          <Camera size={14} />
        </button>

        {/* Duration Badge on Bottom-Right */}
        {item.duration && (
          <div className='absolute bottom-2 right-2 bg-black/85 px-2 py-0.5 rounded text-[11px] font-mono font-bold text-white tracking-wider backdrop-blur-xs z-10'>
            {item.duration}
          </div>
        )}

        {/* Site Badge on Bottom-Left */}
        <div
          style={{ borderColor: item.siteColor }}
          className='absolute bottom-2 left-2 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-black/85 text-white border z-10 backdrop-blur-xs'
        >
          {item.siteName}
        </div>
      </div>

      {/* Video Card Content */}
      <div className='p-3.5 flex-grow flex flex-col justify-between'>
        <div>
          <h3
            className='text-xs sm:text-sm font-semibold line-clamp-2 text-[#e0e0ff] group-hover:text-white transition-colors'
            title={item.title}
          >
            {item.title}
          </h3>
          <div className='flex items-center gap-2 text-xs text-[#6a6a8a] mt-1'>
            {item.views && (
              <span className='flex items-center gap-1'>
                <Eye size={11} /> {item.views}
              </span>
            )}
            {item.duration && (
              <span className='flex items-center gap-1 font-mono'>
                <Clock size={11} /> {item.duration}
              </span>
            )}
          </div>
        </div>

        {/* Action Row */}
        <div className='flex items-center gap-2 mt-3 pt-2.5 border-t border-[#1a1a2e]'>
          {/* Quick Teaser Preview Modal Button */}
          {hasVideo && (
            <button
              type='button'
              aria-label='Open interactive video teaser'
              onClick={e => {
                e.stopPropagation();
                onOpenTeaser(item);
              }}
              title='Open interactive video teaser'
              className='py-1.5 px-2 rounded-md text-xs font-semibold bg-[#05d9e8]/15 hover:bg-[#05d9e8] text-[#05d9e8] hover:text-black border border-[#05d9e8]/40 transition-all flex items-center gap-1 cursor-pointer'
            >
              <Play size={11} className='fill-current' />
            </button>
          )}

          {collections.length > 0 && (
            <select
              aria-label={`Add ${item.title} to a collection`}
              defaultValue=''
              onClick={event => event.stopPropagation()}
              onChange={event => {
                const collectionId = event.target.value;
                if (collectionId) onAddToCollection(collectionId, item);
                event.target.value = '';
              }}
              className='max-w-24 h-8 bg-[#141424] border border-[#1a1a2e] rounded-md px-1 text-[10px] text-[#a0a0cf] focus:border-[#05d9e8] outline-none'
            >
              <option value='' disabled>+ LIST</option>
              {collections.map(collection => (
                <option key={collection.id} value={collection.id}>{collection.name}</option>
              ))}
            </select>
          )}

          {/* Share Button */}
          <button
            type='button'
            aria-label='Share video link'
            onClick={e => {
              e.stopPropagation();
              onShare(item);
            }}
            title='Share video'
            className='w-8 h-8 rounded bg-[#1a1a2e] border border-[#1a1a2e]/50 text-[#a0a0cf] hover:bg-[#ff2a6d]/20 hover:text-[#ff2a6d] hover:border-[#ff2a6d]/50 transition-all flex items-center justify-center active:scale-95 cursor-pointer'
          >
            <Share2 size={14} strokeWidth={2} />
          </button>

          {/* Copy Link Button */}
          <button
            onClick={handleCopy}
            title='Copy video link'
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              isCopied
                ? 'bg-emerald-600 text-white shadow-[0_0_10px_rgba(16,185,129,0.5)]'
                : 'bg-[#141424] hover:bg-[#ff2a6d]/20 hover:text-[#ff2a6d] border border-[#1a1a2e] hover:border-[#ff2a6d]/50 text-[#a0a0cf]'
            }`}
          >
            {isCopied ? (
              <>
                <Check size={13} />
                <span>COPIED!</span>
              </>
            ) : (
              <>
                <Copy size={13} />
                <span>COPY LINK</span>
              </>
            )}
          </button>

          {/* Watch Link */}
          <a
            href={item.url}
            target='_blank'
            rel='noopener noreferrer'
            onClick={event => {
              event.stopPropagation();
              onMarkWatched(item);
            }}
            className='flex items-center justify-center gap-1 py-1.5 px-3 rounded-md text-xs font-semibold bg-[#ff2a6d]/15 hover:bg-[#ff2a6d] text-[#ff2a6d] hover:text-white border border-[#ff2a6d]/40 transition-all cursor-pointer'
          >
            <span>WATCH</span>
            <ExternalLink size={12} />
          </a>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [activeTab, setActiveTab] = useState<
    'hunt' | 'fav' | 'test' | 'screenshots' | 'recent' | 'collections' | 'import-export'
  >('hunt');
  const [query, setQuery] = useState('');
  const [site, setSite] = useState<string>('xvideos');
  const [page, setPage] = useState(1);
  const [results, setResults] = useState<VideoResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Trending suggestions state
  const [selectedTrendingCategory, setSelectedTrendingCategory] =
    useState<TrendingCategory | null>(null);
  const [trendingExpanded, setTrendingExpanded] = useState(false);

  // Active video preview modal
  const [previewModalItem, setPreviewModalItem] = useState<VideoResult | null>(
    null
  );
  const [modalMuted, setModalMuted] = useState(true);
  const [modalVideoSrc, setModalVideoSrc] = useState<string>('');
  const [modalFullVideoSrc, setModalFullVideoSrc] = useState<string>('');
  const [modalUsingFull, setModalUsingFull] = useState(false);
  const [modalVideoLoaded, setModalVideoLoaded] = useState(false);
  const [modalVideoRetried, setModalVideoRetried] = useState(false);
  const [modalFetchingFull, setModalFetchingFull] = useState(false);

  const handleModalVideoError = useCallback(() => {
    if (modalVideoRetried || !modalVideoSrc || modalVideoSrc.startsWith('/api/proxy-media')) {
      return;
    }
    setModalVideoRetried(true);
    setModalVideoLoaded(false);
    setModalVideoSrc(`/api/proxy-media?url=${encodeURIComponent(modalVideoSrc)}`);
  }, [modalVideoRetried, modalVideoSrc]);

  // PWA install help modal.
  const [showInstallModal, setShowInstallModal] = useState(false);

  // Progressive web app install hook
  const { isInstallable, isInstalled, isIOS, isAndroid, install } =
    usePWAInstall();

  // Favorites state
  const [favorites, setFavorites] = useState<VideoResult[]>([]);

  // Saved video screenshots state
  const [screenshots, setScreenshots] = useState<SavedScreenshot[]>([]);

  // Search history state
  const [history, setHistory] = useState<string[]>([]);

  // Recently watched history state
  const [recentlyWatched, setRecentlyWatched] = useState<RecentlyWatchedItem[]>([]);

  // Collections / Playlists state
  const [collections, setCollections] = useState<Collection[]>([]);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [newCollectionColor, setNewCollectionColor] = useState('#ff2a6d');

  // Export / Import state
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);

  // Site tester state
  const [siteTests, setSiteTests] = useState<SiteTestResult[]>([]);
  const [testingSites, setTestingSites] = useState(false);
  const [testFilter, setTestFilter] = useState<'all' | 'online' | 'error'>(
    'all'
  );

  // Console logs
  const [logs, setLogs] = useState<string[]>([]);
  const [consoleOpen, setConsoleOpen] = useState(true);
  const consoleBottomRef = useRef<HTMLDivElement>(null);

  // Track the search context independently of the currently edited form fields.
  const activeSearchRef = useRef({ query: '', site: 'xvideos' });
  const seenUrls = useRef<Set<string>>(new Set());

  // Prevent duplicate concurrent requests when scrolling fast
  const loadingRef = useRef(false);

  const addLog = useCallback((msg: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-80), `[${time}] ${msg}`]);
  }, []);

  // Show transient toast notification
  const showToast = useCallback((msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(curr => (curr === msg ? null : curr));
    }, 2800);
  }, []);

  // Load and validate persisted data before it reaches links, images, or CSS.
  useEffect(() => {
    const readArray = (key: string): unknown[] => {
      try {
        const value: unknown = JSON.parse(localStorage.getItem(key) || '[]');
        return Array.isArray(value) ? value : [];
      } catch {
        return [];
      }
    };

    setFavorites(readArray('pyrmsearchFavorites').map(sanitizeVideoResult).filter((item): item is VideoResult => item !== null));
    setScreenshots(readArray('pyrmsearchScreenshots').map(sanitizeScreenshot).filter((item): item is SavedScreenshot => item !== null));
    setHistory(sanitizeHistory(readArray('searchHistory')));
    setRecentlyWatched(readArray('pyrmsearchRecentlyWatched').map(sanitizeRecentlyWatched).filter((item): item is RecentlyWatchedItem => item !== null));
    setCollections(readArray('pyrmsearchCollections').map(sanitizeCollection).filter((item): item is Collection => item !== null));

    addLog(`Pyrmsearch ready • ${Object.keys(SITES).length} public sources configured`);
  }, [addLog]);

  // Scroll console to bottom on new log
  useEffect(() => {
    if (consoleOpen && consoleBottomRef.current) {
      consoleBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, consoleOpen]);

  // Copy links only when the browser confirms the operation.
  const handleCopyLink = useCallback(
    async (e: React.MouseEvent, item: VideoResult): Promise<boolean> => {
      e.stopPropagation();
      let copied = false;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(item.url);
          copied = true;
        }
      } catch {
        // Try the legacy clipboard API on older mobile browsers.
      }

      if (!copied) {
        const input = document.createElement('textarea');
        input.value = item.url;
        input.setAttribute('readonly', '');
        input.style.position = 'fixed';
        input.style.opacity = '0';
        input.style.pointerEvents = 'none';
        document.body.appendChild(input);
        input.select();
        try {
          copied = document.execCommand('copy');
        } catch {
          copied = false;
        } finally {
          input.remove();
        }
      }

      showToast(copied ? 'Link copied to clipboard.' : 'Clipboard access was denied.');
      if (copied) addLog(`Copied link: ${item.title.substring(0, 36)}…`);
      return copied;
    },
    [addLog, showToast]
  );

  // Toggle favorite
  const handleToggleFavorite = useCallback(
    (e: React.MouseEvent, item: VideoResult) => {
      e.stopPropagation();
      setFavorites(prev => {
        const exists = prev.some(f => f.url === item.url);
        let updated: VideoResult[];
        if (exists) {
          updated = prev.filter(f => f.url !== item.url);
          addLog(`Removed favorite: ${item.title.substring(0, 36)}...`);
          showToast('Removed from favorites');
        } else {
          updated = [item, ...prev];
          addLog(`🔞🔞 Saved favorite: ${item.title.substring(0, 36)}...`);
          showToast('Saved to favorites!');
        }
        try {
          localStorage.setItem('pyrmsearchFavorites', JSON.stringify(updated));
        } catch (err) {
          console.error('Storage quota error', err);
          showToast('Browser storage is full; this favorite may not persist.');
        }
        return updated;
      });
    },
    [addLog, showToast]
  );

  // Capture a compressed frame. Cross-origin streams can taint canvases, so fail gracefully.
  const captureVideoFrame = useCallback(
    (videoEl: HTMLVideoElement | null): string | null => {
      if (!videoEl || videoEl.videoWidth === 0 || videoEl.videoHeight === 0) return null;
      const scale = Math.min(1, 960 / videoEl.videoWidth, 540 / videoEl.videoHeight);
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(videoEl.videoWidth * scale));
      canvas.height = Math.max(1, Math.round(videoEl.videoHeight * scale));
      const context = canvas.getContext('2d');
      if (!context) return null;
      try {
        context.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.78);
      } catch {
        return null;
      }
    },
    []
  );

  // Save a screenshot to localStorage with a conservative cap for mobile storage.
  const handleSaveScreenshot = useCallback(
    (item: VideoResult, video: HTMLVideoElement | null) => {
      const dataUrl = captureVideoFrame(video);
      if (!dataUrl) {
        showToast('Preview must be playing before a screenshot can be saved.');
        return;
      }
      if (screenshots.some(screenshot => screenshot.id === item.id)) {
        showToast('A screenshot is already saved for this video.');
        return;
      }

      const entry: SavedScreenshot = {
        id: item.id,
        title: item.title,
        siteName: item.siteName,
        siteColor: item.siteColor,
        thumb: item.thumb,
        dataUrl,
        capturedAt: Date.now()
      };
      const updated = [entry, ...screenshots].slice(0, 50);
      try {
        localStorage.setItem('pyrmsearchScreenshots', JSON.stringify(updated));
      } catch {
        showToast('Browser storage is full; export data or remove screenshots.');
        return;
      }
      setScreenshots(updated);
      addLog(`Screenshot saved: ${item.title.substring(0, 36)}…`);
      showToast('Screenshot saved.');
    },
    [addLog, captureVideoFrame, screenshots, showToast]
  );

  // Delete a saved screenshot.
  const handleDeleteScreenshot = useCallback(
    (id: string) => {
      const updated = screenshots.filter(screenshot => screenshot.id !== id);
      setScreenshots(updated);
      try {
        localStorage.setItem('pyrmsearchScreenshots', JSON.stringify(updated));
      } catch {
        // The in-memory state is still updated if local storage is unavailable.
      }
      addLog('Screenshot deleted');
      showToast('Screenshot deleted');
    },
    [addLog, screenshots, showToast]
  );

  // Clear all screenshots
  const handleClearScreenshots = () => {
    if (screenshots.length === 0) return;
    if (!window.confirm('Delete all saved screenshots?')) return;
    setScreenshots([]);
    try {
      localStorage.removeItem('pyrmsearchScreenshots');
    } catch {
      // ignore
    }
    addLog('All screenshots cleared');
    showToast('Screenshots cleared');
  };

  const favoritedUrls = useMemo(
    () => new Set(favorites.map(f => f.url)),
    [favorites]
  );

  const isFavorited = useCallback(
    (url: string) => favoritedUrls.has(url),
    [favoritedUrls]
  );

  const handleClearFavorites = () => {
    if (favorites.length === 0) return;
    if (!window.confirm('Wipe all favorites from local storage?')) return;
    setFavorites([]);
    try {
      localStorage.removeItem('pyrmsearchFavorites');
    } catch {
      // ignore
    }
    addLog('All favorites cleared');
    showToast('Favorites cleared');
  };

  // Add item to recently watched history.
  const addToRecentlyWatched = useCallback((item: VideoResult) => {
    setRecentlyWatched(previous => {
      const entry: RecentlyWatchedItem = { ...item, watchedAt: Date.now() };
      const updated = [entry, ...previous.filter(recent => recent.url !== item.url)].slice(0, 50);
      try {
        localStorage.setItem('pyrmsearchRecentlyWatched', JSON.stringify(updated));
      } catch {
        // Keep the in-memory history if storage is unavailable or full.
      }
      return updated;
    });
  }, []);

  // Load the teaser first, then ask the server for a same-site full-video source.
  useEffect(() => {
    let cancelled = false;

    if (!previewModalItem) {
      setModalVideoSrc('');
      setModalFullVideoSrc('');
      setModalUsingFull(false);
      setModalVideoLoaded(false);
      setModalVideoRetried(false);
      setModalFetchingFull(false);
      return;
    }

    addToRecentlyWatched(previewModalItem);
    setModalVideoLoaded(false);
    setModalVideoRetried(false);
    setModalMuted(true);
    setModalFullVideoSrc('');
    setModalFetchingFull(true);
    setModalUsingFull(false);
    setModalVideoSrc(previewModalItem.previewVideo || '');

    void fetchFullVideoUrl(previewModalItem.url, previewModalItem.site).then(fullUrl => {
      if (cancelled || !fullUrl) return;
      setModalFullVideoSrc(fullUrl);
      if (!previewModalItem.previewVideo) {
        setModalVideoSrc(fullUrl);
        setModalUsingFull(true);
      }
    }).finally(() => {
      if (!cancelled) setModalFetchingFull(false);
    });

    return () => {
      cancelled = true;
    };
  }, [previewModalItem, addToRecentlyWatched]);

  // Create a new collection
  const handleCreateCollection = () => {
    const name = newCollectionName.trim();
    if (!name) {
      showToast('Please enter a collection name');
      return;
    }
    if (collections.some(c => c.name.toLowerCase() === name.toLowerCase())) {
      showToast('Collection already exists');
      return;
    }
    const collection: Collection = {
      id: Date.now().toString(),
      name,
      color: newCollectionColor,
      videoIds: [],
      videos: [],
      createdAt: Date.now()
    };
    setCollections(prev => {
      const updated = [collection, ...prev];
      try {
        localStorage.setItem('pyrmsearchCollections', JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
    setNewCollectionName('');
    showToast(`Collection "${name}" created`);
  };

  // Delete a collection
  const handleDeleteCollection = (collectionId: string) => {
    if (!window.confirm('Delete this collection?')) return;
    setCollections(prev => {
      const updated = prev.filter(c => c.id !== collectionId);
      try {
        localStorage.setItem('pyrmsearchCollections', JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
    showToast('Collection deleted');
  };

  const handleClearCollections = () => {
    if (collections.length === 0 || !window.confirm('Delete all collections?')) return;
    setCollections([]);
    try {
      localStorage.removeItem('pyrmsearchCollections');
    } catch {
      // The in-memory state is still cleared.
    }
    showToast('All collections cleared.');
  };

  // Store the full safe result with the collection so it remains visible after a new search.
  const handleAddToCollection = (collectionId: string, item: VideoResult) => {
    const collection = collections.find(candidate => candidate.id === collectionId);
    if (!collection) return;
    if (collection.videoIds.includes(item.url)) {
      showToast('That video is already in this collection.');
      return;
    }
    const updated = collections.map(candidate => candidate.id !== collectionId
      ? candidate
      : {
          ...candidate,
          videoIds: [...candidate.videoIds, item.url],
          videos: [...candidate.videos.filter(video => video.url !== item.url), item]
        }
    );
    setCollections(updated);
    try {
      localStorage.setItem('pyrmsearchCollections', JSON.stringify(updated));
    } catch {
      showToast('Browser storage is full; the collection was not saved.');
      return;
    }
    showToast(`Added to ${collection.name}.`);
  };

  // Remove a result from a collection.
  const handleRemoveFromCollection = (collectionId: string, videoUrl: string) => {
    const updated = collections.map(collection => collection.id !== collectionId
      ? collection
      : {
          ...collection,
          videoIds: collection.videoIds.filter(id => id !== videoUrl),
          videos: collection.videos.filter(video => video.url !== videoUrl)
        }
    );
    setCollections(updated);
    try {
      localStorage.setItem('pyrmsearchCollections', JSON.stringify(updated));
    } catch {
      // Keep the change in memory if storage is unavailable.
    }
    showToast('Removed from collection.');
  };

  // Export private browser data to a local JSON backup.
  const handleExportData = async () => {
    setIsExporting(true);
    try {
      const data = {
        favorites,
        screenshots,
        history,
        recentlyWatched,
        collections,
        exportedAt: Date.now()
      };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `pyrmsearch-backup-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      showToast('Backup downloaded.');
      addLog('Exported local Pyrmsearch backup');
    } catch {
      showToast('Could not create the backup file.');
    } finally {
      setIsExporting(false);
    }
  };

  // Import a size-limited, validated backup and only update app state after storage succeeds.
  const handleImportData = async (file: File) => {
    setIsImporting(true);
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error('Backup file is too large (20 MB maximum).');
      const data = parseBackupText(await file.text());
      const updates: Record<string, string> = {};
      if (data.favorites) updates.pyrmsearchFavorites = JSON.stringify(data.favorites);
      if (data.screenshots) updates.pyrmsearchScreenshots = JSON.stringify(data.screenshots);
      if (data.history) updates.searchHistory = JSON.stringify(data.history);
      if (data.recentlyWatched) updates.pyrmsearchRecentlyWatched = JSON.stringify(data.recentlyWatched);
      if (data.collections) updates.pyrmsearchCollections = JSON.stringify(data.collections);

      const originalValues = new Map<string, string | null>();
      try {
        for (const [key, value] of Object.entries(updates)) {
          originalValues.set(key, localStorage.getItem(key));
          localStorage.setItem(key, value);
        }
      } catch {
        for (const [key, value] of originalValues) {
          try {
            if (value === null) localStorage.removeItem(key);
            else localStorage.setItem(key, value);
          } catch {
            // Best-effort rollback if the storage quota is exhausted.
          }
        }
        throw new Error('Not enough browser storage for this backup. Remove some screenshots and try again.');
      }

      if (data.favorites) setFavorites(data.favorites);
      if (data.screenshots) setScreenshots(data.screenshots);
      if (data.history) setHistory(data.history);
      if (data.recentlyWatched) setRecentlyWatched(data.recentlyWatched);
      if (data.collections) setCollections(data.collections);
      setImportFile(null);
      showToast('Backup imported. Unsafe or unsupported entries were skipped.');
      addLog('Imported validated local Pyrmsearch backup');
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Import failed.');
    } finally {
      setIsImporting(false);
    }
  };

  const handleShare = async (item: VideoResult) => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: item.title,
          text: `Open this listing on ${item.siteName}.`,
          url: item.url
        });
        return;
      }
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(item.url);
      showToast('Link copied to clipboard.');
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      showToast('Sharing is unavailable in this browser.');
    }
  };

  const handleInstallApp = async () => {
    const success = await install();
    if (success) {
      showToast('Pyrmsearch installed.');
      addLog('Pyrmsearch installed to the device home screen');
      setShowInstallModal(false);
    }
  };

  // Execute Search / Hunt
  const handleExecute = async (
    targetPage: number = 1,
    overrideQuery?: string,
    overrideSite?: string
  ) => {
    if (loadingRef.current) return;

    const q = (
      overrideQuery !== undefined
        ? overrideQuery
        : targetPage === 1
          ? query
          : activeSearchRef.current.query
    ).trim();
    const currentSite = overrideSite || (targetPage === 1 ? site : activeSearchRef.current.site);
    if (overrideSite && overrideSite !== site) {
      setSite(overrideSite);
      setPage(1);
      seenUrls.current.clear();
      setHasMore(true);
    }

    const cfg: SiteConfig = SITES[currentSite] || SITES.xvideos;

    if (!q) {
      showToast('Please enter a search query.');
      return;
    }
    if (q.length > 120) {
      showToast('Search queries are limited to 120 characters.');
      return;
    }
    if (!SITES[currentSite]) {
      showToast('Choose a supported source site.');
      return;
    }
    if (targetPage === 1) {
      activeSearchRef.current = { query: q, site: currentSite };
      setHasSearched(true);
    }

    loadingRef.current = true;
    setLoading(true);

    if (targetPage === 1) {
      seenUrls.current.clear();
      setResults([]);
      setHasMore(true);
      setActiveTab('hunt');

      // Update search history
      setHistory(prev => {
        const filtered = prev.filter(
          item => item.toLowerCase() !== q.toLowerCase()
        );
        const updated = [q, ...filtered.slice(0, 11)];
        try {
          localStorage.setItem('searchHistory', JSON.stringify(updated));
        } catch {
          // ignore
        }
        return updated;
      });
    }

    addLog(`🔞 Hunting ${cfg.name} (page ${targetPage}) for "${q}"...`);

    try {
      const data = await scrapeSite(currentSite, q, targetPage);

      // Filter out duplicate URLs
      const newItems = data.filter(item => {
        if (!item.url || seenUrls.current.has(item.url)) return false;
        seenUrls.current.add(item.url);
        return true;
      });

      if (targetPage === 1) {
        setResults(newItems);
      } else {
        setResults(prev => [...prev, ...newItems]);
      }

      setPage(targetPage);

      const previewCount = newItems.filter(i => Boolean(i.previewVideo)).length;

      if (newItems.length === 0) {
        setHasMore(false);
        addLog(`Page ${targetPage}: 0 new results • end of stream`);
      } else {
        addLog(
          `🔞 Page ${targetPage}: +${newItems.length} videos (${previewCount} with hover video preview)`
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      console.error(error);
      addLog(`Search failed: ${message}`);
      showToast(message);
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  };

  // Infinite Scroll listener with loadingRef guard
  useEffect(() => {
    const handleScroll = () => {
      if (
        activeTab !== 'hunt' ||
        loadingRef.current ||
        !hasMore ||
        results.length === 0
      )
        return;
      const scrollPos = window.innerHeight + window.scrollY;
      const threshold = document.body.offsetHeight - 500;
      if (scrollPos >= threshold) {
        handleExecute(page + 1);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [activeTab, hasMore, results.length, page]);

  // Run Site Health Tests for all sites
  const handleRunSiteTests = async () => {
    setTestingSites(true);
    addLog(
      `⚡ Testing connectivity and latency for all ${Object.keys(SITES).length} sites...`
    );
    try {
      const testResults = await testAllSites();
      setSiteTests(testResults);
      const onlineCount = testResults.filter(s => s.ok).length;
      addLog(
        `⚡ Tests completed: ${onlineCount}/${testResults.length} sites operational`
      );
      showToast(`Tests finished: ${onlineCount}/${testResults.length} online`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown diagnostics error.';
      addLog(`Site tests failed: ${message}`);
      showToast(message);
    } finally {
      setTestingSites(false);
    }
  };

  // Test one site without downloading and parsing an entire search result page.
  const handleTestSingleSite = async (siteId: string) => {
    const config = SITES[siteId];
    if (!config) return;
    addLog(`Checking ${config.name} connectivity…`);
    const startedAt = Date.now();
    try {
      const [result] = await testAllSites(siteId);
      if (!result) throw new Error('No diagnostic result was returned.');
      setSiteTests(previous => [...previous.filter(item => item.id !== siteId), result]);
      addLog(`${config.name}: ${result.ok ? 'reachable' : 'unavailable'} (${result.latencyMs}ms)`);
      showToast(`${config.name}: ${result.ok ? 'Online' : 'Unavailable'} (${result.latencyMs}ms)`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown diagnostics error.';
      const result: SiteTestResult = {
        id: config.id,
        name: config.name,
        base: config.base,
        icon: config.icon,
        color: config.color,
        status: 0,
        ok: false,
        latencyMs: Date.now() - startedAt,
        error: message
      };
      setSiteTests(previous => [...previous.filter(item => item.id !== siteId), result]);
      addLog(`${config.name} check failed: ${message}`);
      showToast(message);
    }
  };

  const currentSiteConfig = SITES[site] || SITES.xvideos;
  const onlineSiteCount = siteTests.filter(s => s.ok).length;

  return (
    <div className='min-h-screen text-[#e0e0ff] p-3 sm:p-6 pb-40'>
      {/* Toast Notification */}
      {toastMessage && (
        <div className='fixed top-5 right-5 z-50 flex items-center gap-2 bg-[#ff2a6d] text-white px-4 py-2.5 rounded-lg shadow-[0_0_20px_rgba(255,42,109,0.7)] text-sm font-semibold tracking-wide animate-bounce'>
          <Sparkles size={16} />
          {toastMessage}
        </div>
      )}

      {/* Progressive web app install help */}
      {showInstallModal && (
        <div
          role='presentation'
          onClick={() => setShowInstallModal(false)}
          className='fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4'
        >
          <section
            role='dialog'
            aria-modal='true'
            aria-labelledby='install-title'
            onClick={event => event.stopPropagation()}
            className='bg-[#0a0a14] border border-[#05d9e8] rounded-2xl max-w-lg w-full overflow-hidden shadow-[0_0_35px_rgba(5,217,232,0.35)]'
          >
            <header className='p-4 bg-[#070710] border-b border-[#1a1a2e] flex items-center justify-between'>
              <div className='flex items-center gap-2.5'>
                <div className='w-8 h-8 rounded-lg bg-gradient-to-br from-[#ff2a6d] to-[#05d9e8] flex items-center justify-center text-white text-xs font-black'>PY</div>
                <div>
                  <h2 id='install-title' className='text-sm font-extrabold text-white tracking-wider'>Install Pyrmsearch</h2>
                  <p className='text-[10px] text-[#6a6a8a]'>Progressive web app</p>
                </div>
              </div>
              <button
                type='button'
                aria-label='Close install help'
                onClick={() => setShowInstallModal(false)}
                className='text-[#6a6a8a] hover:text-white p-1 rounded-lg hover:bg-white/10 transition-colors'
              >
                <X size={18} />
              </button>
            </header>

            <div className='p-5 space-y-4'>
              {isInstalled ? (
                <div className='flex items-center gap-2 text-sm font-semibold text-emerald-400'>
                  <CheckCircle2 size={18} /> Installed on this device.
                </div>
              ) : isInstallable ? (
                <button
                  type='button'
                  onClick={handleInstallApp}
                  className='w-full py-3 px-4 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white font-extrabold text-xs tracking-wider flex items-center justify-center gap-2'
                >
                  <Smartphone size={16} /> INSTALL PYRMSEARCH
                </button>
              ) : (
                <div className='text-sm text-[#a0a0cf] bg-[#0c0c1a] p-3 rounded-lg border border-[#1a1a2e]'>
                  {isIOS
                    ? 'On iPhone or iPad, open this page in Safari, tap Share, then Add to Home Screen.'
                    : isAndroid
                      ? 'In Android Chrome, open the browser menu and choose Install app or Add to Home screen.'
                      : 'Use your browser menu and choose Install app or Add to Home screen when that option is available.'}
                </div>
              )}
              <p className='text-xs text-[#6a6a8a] leading-relaxed'>
                The app shell and interface can be cached for quick reopening. Searches and source media still require a network connection. No standalone APK is bundled with this project.
              </p>
            </div>
          </section>
        </div>
      )}

      {/* Video Preview Modal */}
      {previewModalItem && (
        <div
          onClick={() => setPreviewModalItem(null)}
          className='fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-200'
        >
          <div
            onClick={e => e.stopPropagation()}
            className='bg-[#0a0a14] border border-[#ff2a6d] rounded-2xl max-w-2xl w-full overflow-hidden shadow-[0_0_35px_rgba(255,42,109,0.35)] relative'
          >
            {/* Modal Header */}
            <div className='p-4 bg-[#070710] border-b border-[#1a1a2e] flex items-center justify-between'>
              <div className='flex items-center gap-2'>
                <span
                  style={{ backgroundColor: previewModalItem.siteColor }}
                  className='px-2 py-0.5 rounded text-[11px] font-black text-white'
                >
                  {previewModalItem.siteName}
                </span>
                <span className='text-xs text-[#05d9e8] font-bold tracking-wider'>
                  ⚡ FULL VIDEO TEASER PREVIEW
                </span>
              </div>
              <button
                onClick={() => setPreviewModalItem(null)}
                className='text-[#6a6a8a] hover:text-white p-1 rounded-lg hover:bg-white/10 transition-colors cursor-pointer'
              >
                <X size={18} />
              </button>
            </div>

            {/* Video Player */}
            <div className='relative bg-black aspect-video flex items-center justify-center overflow-hidden'>
              {/* Full video fetch status */}
              {modalFetchingFull && !modalVideoSrc && (
                <div className='absolute inset-0 flex items-center justify-center bg-black/60 pointer-events-none z-20'>
                  <div className='flex items-center gap-2 text-xs text-[#05d9e8] font-bold bg-black/80 px-3 py-1.5 rounded-full border border-[#05d9e8]/40 shadow-[0_0_12px_rgba(5,217,232,0.4)]'>
                    <div className='w-3.5 h-3.5 border-2 border-[#05d9e8] border-t-transparent rounded-full animate-spin' />
                    <span>FETCHING FULL VIDEO...</span>
                  </div>
                </div>
              )}

              {modalVideoSrc ? (
                <>
                  <video
                    key={modalVideoSrc}
                    src={modalVideoSrc}
                    autoPlay={!modalUsingFull}
                    controls
                    loop={!modalUsingFull}
                    muted={modalMuted}
                    playsInline
                    referrerPolicy='no-referrer'
                    onLoadedData={() => setModalVideoLoaded(true)}
                    onCanPlay={() => setModalVideoLoaded(true)}
                    onError={handleModalVideoError}
                    className='w-full h-full object-contain'
                  />
                  {!modalVideoLoaded && (
                    <div className='absolute inset-0 flex items-center justify-center bg-black/60 pointer-events-none z-10'>
                      <div className='flex items-center gap-2 text-xs text-[#05d9e8] font-bold bg-black/80 px-3 py-1.5 rounded-full border border-[#05d9e8]/40 shadow-[0_0_12px_rgba(5,217,232,0.4)]'>
                        <div className='w-3.5 h-3.5 border-2 border-[#05d9e8] border-t-transparent rounded-full animate-spin' />
                        <span>
                          {modalUsingFull
                            ? 'BUFFERING FULL VIDEO...'
                            : 'BUFFERING PREVIEW...'}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Full video toggle badge */}
                  {modalFullVideoSrc && !modalUsingFull && (
                    <button
                      onClick={() => {
                        setModalVideoSrc(modalFullVideoSrc);
                        setModalUsingFull(true);
                        setModalVideoLoaded(false);
                        setModalVideoRetried(false);
                        setModalMuted(false);
                      }}
                      className='absolute bottom-4 left-4 bg-[#05d9e8] text-black px-3 py-1.5 rounded-full text-xs font-black tracking-wider shadow-[0_0_15px_rgba(5,217,232,0.6)] hover:opacity-90 transition-all cursor-pointer z-20 flex items-center gap-1.5'
                    >
                      <Play size={12} className='fill-current' />
                      <span>PLAY FULL VIDEO</span>
                    </button>
                  )}

                  {/* Back to preview badge */}
                  {modalUsingFull && previewModalItem.previewVideo && (
                    <button
                      onClick={() => {
                        setModalVideoSrc(previewModalItem.previewVideo || '');
                        setModalUsingFull(false);
                        setModalVideoLoaded(false);
                        setModalVideoRetried(false);
                        setModalMuted(true);
                      }}
                      className='absolute bottom-4 left-4 bg-[#ff2a6d] text-white px-3 py-1.5 rounded-full text-xs font-black tracking-wider shadow-[0_0_15px_rgba(255,42,109,0.6)] hover:opacity-90 transition-all cursor-pointer z-20 flex items-center gap-1.5'
                    >
                      <Film size={12} className='fill-current' />
                      <span>PREVIEW CLIP</span>
                    </button>
                  )}

                  {/* Sound Toggle */}
                  <button
                    onClick={() => setModalMuted(!modalMuted)}
                    className='absolute bottom-4 right-4 bg-black/80 p-2 rounded-full border border-white/20 text-white hover:bg-[#ff2a6d] transition-all cursor-pointer z-20'
                  >
                    {modalMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                  </button>
                </>
              ) : (
                <div className='text-center p-6'>
                  {previewModalItem.thumb && (
                    <img
                      src={previewModalItem.thumb}
                      alt={previewModalItem.title}
                      referrerPolicy='no-referrer'
                      className='max-h-60 mx-auto rounded-lg mb-3 object-cover'
                    />
                  )}
                  <p className='text-xs text-[#6a6a8a]'>
                    Hover video clip not provided by source tube. Click below to
                    watch the full video on {previewModalItem.siteName}.
                  </p>
                </div>
              )}
            </div>

            {/* Modal Body Info */}
            <div className='p-4 sm:p-5'>
              <h3 className='text-base font-bold text-white mb-2 line-clamp-2'>
                {previewModalItem.title}
              </h3>

              <div className='flex items-center gap-4 text-xs text-[#6a6a8a] mb-4'>
                {previewModalItem.duration && (
                  <span className='flex items-center gap-1 text-[#e0e0ff]'>
                    <Clock size={13} /> {previewModalItem.duration}
                  </span>
                )}
                {previewModalItem.views && (
                  <span className='flex items-center gap-1 text-[#e0e0ff]'>
                    <Eye size={13} /> {previewModalItem.views}
                  </span>
                )}
                <span className='font-mono text-[#05d9e8]'>
                  {previewModalItem.siteName}
                </span>
              </div>

              {/* Action Buttons */}
              <div className='flex items-center gap-3 pt-3 border-t border-[#1a1a2e]'>
                <button
                  onClick={e => handleCopyLink(e, previewModalItem)}
                  className='flex-1 py-2.5 px-4 rounded-lg bg-[#141424] hover:bg-[#ff2a6d]/20 text-white font-semibold text-xs border border-[#1a1a2e] hover:border-[#ff2a6d] transition-all flex items-center justify-center gap-2 cursor-pointer'
                >
                  <Copy size={14} />
                  <span>COPY VIDEO LINK</span>
                </button>

                <button
                  onClick={e => handleToggleFavorite(e, previewModalItem)}
                  className={`py-2.5 px-4 rounded-lg font-semibold text-xs transition-all flex items-center justify-center gap-2 cursor-pointer ${
                    isFavorited(previewModalItem.url)
                      ? 'bg-[#ff2a6d] text-white shadow-[0_0_12px_#ff2a6d]'
                      : 'bg-[#141424] border border-[#1a1a2e] text-white hover:bg-[#ff2a6d]/20'
                  }`}
                >
                  <Heart
                    size={14}
                    className={
                      isFavorited(previewModalItem.url) ? 'fill-current' : ''
                    }
                  />
                  <span>
                    {isFavorited(previewModalItem.url) ? 'SAVED' : 'SAVE FAV'}
                  </span>
                </button>

                <a
                  href={previewModalItem.url}
                  target='_blank'
                  rel='noopener noreferrer'
                  className='py-2.5 px-6 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white font-extrabold text-xs tracking-wider shadow-[0_0_15px_rgba(255,42,109,0.4)] flex items-center justify-center gap-2 hover:opacity-95 cursor-pointer'
                >
                  <span>
                    WATCH ON {previewModalItem.siteName.toUpperCase()}
                  </span>
                  <ExternalLink size={14} />
                </a>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-6 mb-6 shadow-[0_0_20px_rgba(255,42,109,0.2)]'>
        <div className='flex items-center justify-center gap-2'>
          <h1 className='text-3xl sm:text-4xl font-black tracking-tighter bg-gradient-to-r from-[#ff2a6d] via-[#d300c5] to-[#05d9e8] bg-clip-text text-transparent'>
            pyrmsearch
          </h1>
        </div>
      </header>

      <aside role='note' className='mb-5 flex items-start gap-2 rounded-lg border border-[#1a1a2e] bg-[#0a0a12]/80 px-3 py-2.5 text-xs text-[#a0a0cf]'>
        <ShieldCheck size={15} className='mt-0.5 shrink-0 text-[#05d9e8]' />
        <span><strong className='text-white'>18+ only.</strong> Searches are sent to the selected source through this server; saved items stay in this browser unless exported. Use public pages only, and follow applicable laws and each source’s terms.</span>
      </aside>

      {/* Navigation Tabs */}
      <div className='flex items-center gap-2 sm:gap-3 mb-5 overflow-x-auto pb-1'>
        <button
          onClick={() => setActiveTab('hunt')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${
            activeTab === 'hunt'
              ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#ff2a6d] text-white shadow-[0_0_15px_rgba(255,42,109,0.4)]'
              : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#05d9e8] hover:text-[#05d9e8]'
          }`}
        >
          <span>🔞</span> HUNT
        </button>

        <button
          onClick={() => setActiveTab('fav')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${
            activeTab === 'fav'
              ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#ff2a6d] text-white shadow-[0_0_15px_rgba(255,42,109,0.4)]'
              : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#05d9e8] hover:text-[#05d9e8]'
          }`}
        >
          <span>🔞🔞</span> FAVORITES
          <span className='text-[#ff2a6d] font-mono text-xs'>
            ({favorites.length})
          </span>
        </button>

        <button
          onClick={() => {
            setActiveTab('test');
            if (siteTests.length === 0) handleRunSiteTests();
          }}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${
            activeTab === 'test'
              ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#05d9e8] text-white shadow-[0_0_15px_rgba(5,217,232,0.4)]'
              : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#05d9e8] hover:text-[#05d9e8]'
          }`}
        >
          <Activity size={15} /> SITE DIAGNOSTICS ({Object.keys(SITES).length})
        </button>

        <button
          onClick={() => setActiveTab('screenshots')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${
            activeTab === 'screenshots'
              ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#05d9e8] text-white shadow-[0_0_15px_rgba(255,42,109,0.4)]'
              : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#05d9e8] hover:text-[#05d9e8]'
          }`}
        >
          <Camera size={15} /> SCREENSHOTS ({screenshots.length})
        </button>

        <button
          onClick={() => setActiveTab('recent')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${activeTab === 'recent' ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#ff2a6d] text-white shadow-[0_0_15px_rgba(255,42,109,0.4)]' : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#ff2a6d] hover:text-[#ff2a6d]'}`}
        >
          <History size={15} /> RECENT ({recentlyWatched.length})
        </button>

        <button
          onClick={() => setActiveTab('collections')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${activeTab === 'collections' ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#05d9e8] text-white shadow-[0_0_15px_rgba(255,42,109,0.4)]' : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#ff2a6d] hover:text-[#ff2a6d]'}`}
        >
          <SaveIcon size={15} /> COLLECTIONS ({collections.length})
        </button>

        <button
          onClick={() => setActiveTab('import-export')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs sm:text-sm font-bold tracking-wider transition-all cursor-pointer ${activeTab === 'import-export' ? 'bg-gradient-to-r from-[#ff2a6d]/30 to-[#05d9e8]/20 border border-[#ff2a6d] text-white shadow-[0_0_15px_rgba(255,42,109,0.4)]' : 'bg-[#0a0a12] border border-[#1a1a2e] text-[#6a6a8a] hover:border-[#ff2a6d] hover:text-[#ff2a6d]'}`}
        >
          <Upload size={15} /> BACKUP
        </button>

        <button
          type='button'
          onClick={() => setShowInstallModal(true)}
          className='ml-auto hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#05d9e8]/10 border border-[#05d9e8]/40 text-[#05d9e8] text-xs font-bold hover:bg-[#05d9e8]/20 transition-all cursor-pointer'
        >
          <Smartphone size={13} />
          <span>INSTALL APP</span>
        </button>
      </div>

      {/* Tab 1: HUNT PANEL */}
      {activeTab === 'hunt' && (
        <div>
          {/* Search Controls */}
          <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-3 sm:p-4 mb-4 shadow-[0_0_12px_rgba(5,217,232,0.1)]'>
            <div className='flex flex-col sm:flex-row gap-2.5'>
              {/* Site Selector with all 13 sites */}
              <div className='relative flex-shrink-0'>
                <select
                  value={site}
                  disabled={loading}
                  onChange={event => {
                    const nextSite = event.target.value;
                    setSite(nextSite);
                    setResults([]);
                    setPage(1);
                    setHasMore(true);
                    setHasSearched(false);
                    activeSearchRef.current = { query: '', site: nextSite };
                    seenUrls.current.clear();
                  }}
                  style={{ borderColor: currentSiteConfig.color }}
                  className='w-full sm:w-48 bg-[#05050a] border text-white rounded-lg px-3.5 py-3 text-sm font-semibold outline-none transition-colors cursor-pointer'
                >
                  <optgroup label='Popular Tubes'>
                    <option value='xvideos'>XVideos (XV)</option>
                    <option value='okxxx'>OK.XXX (OK)</option>
                    <option value='helloporn'>Hello.Porn (HP)</option>
                    <option value='usersporn'>Users.Porn (UP)</option>
                    <option value='porntrex'>PornTrex (PT)</option>
                    <option value='hqporner'>HQPorner (HQ)</option>
                    <option value='xnxx'>XNXX (XN)</option>
                    <option value='eporner'>Eporner (EP)</option>
                    <option value='xhamster'>XHamster (XH)</option>
                    <option value='pornhub'>Pornhub (PH)</option>
                    <option value='redtube'>RedTube (RT)</option>
                    <option value='youporn'>YouPorn (YP)</option>
                    <option value='spankbang'>SpankBang (SB)</option>
                  </optgroup>
                </select>
              </div>

              {/* Search Input */}
              <div className='relative flex-grow'>
                <input
                  type='search'
                  maxLength={120}
                  aria-label='Search public listings'
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleExecute(1)}
                  placeholder='Type search terms (e.g. amateur, blonde, 4k)...'
                  className='w-full bg-[#05050a] border border-[#1a1a2e] focus:border-[#05d9e8] text-white rounded-lg px-4 py-3 text-sm outline-none transition-all focus:shadow-[0_0_10px_rgba(5,217,232,0.3)] placeholder-[#6a6a8a]'
                />
                {query && (
                  <button
                    onClick={() => setQuery('')}
                    className='absolute right-3 top-3 text-[#6a6a8a] hover:text-white text-xs px-1 cursor-pointer'
                  >
                    ✕
                  </button>
                )}
              </div>

              {/* Hunt Button */}
              <button
                onClick={() => handleExecute(1)}
                disabled={loading}
                className='bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] hover:opacity-95 text-white font-extrabold px-6 py-3 rounded-lg text-sm tracking-wider cursor-pointer shadow-[0_0_15px_rgba(255,42,109,0.4)] hover:shadow-[0_0_25px_rgba(255,42,109,0.65)] hover:-translate-y-0.5 transition-all flex items-center justify-center gap-2 flex-shrink-0 disabled:opacity-50'
              >
                <Search size={16} />
                <span>🔞 HUNT</span>
              </button>
            </div>

            {/* Popular Quick Keyword Tags */}
            <div className='flex items-center gap-1.5 flex-wrap mt-3 pt-3 border-t border-[#1a1a2e]'>
              <span className='text-[11px] text-[#6a6a8a] flex items-center gap-1 mr-1'>
                <Flame size={12} className='text-[#ff2a6d]' /> Trending:
              </span>
              {POPULAR_TAGS.map(tag => (
                <button
                  key={tag}
                  onClick={() => {
                    setQuery(tag);
                    handleExecute(1, tag);
                  }}
                  className='px-2.5 py-1 bg-[#05050a] border border-[#1a1a2e] hover:border-[#ff2a6d] rounded text-[11px] text-[#e0e0ff] hover:text-[#ff2a6d] transition-all cursor-pointer'
                >
                  #{tag}
                </button>
              ))}
            </div>

            {/* Search History Chips */}
            {history.length > 0 && (
              <div className='flex items-center gap-1.5 flex-wrap mt-2.5'>
                <span className='text-[11px] text-[#6a6a8a] flex items-center gap-1 mr-1'>
                  <Clock size={12} /> Recent:
                </span>
                {history.slice(0, 8).map((hist, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setQuery(hist);
                      handleExecute(1, hist);
                    }}
                    className='px-2.5 py-0.5 bg-[#121224] border border-[#1a1a2e] hover:border-[#05d9e8] rounded-full text-[11px] text-[#a0a0cf] hover:text-white transition-all cursor-pointer'
                  >
                    {hist}
                  </button>
                ))}
              </div>
            )}

            {/* Trending Suggestions Section */}
            <div className='mt-4 pt-4 border-t border-[#1a1a2e]'>
              <div className='flex items-center justify-between mb-2.5'>
                <span className='text-[11px] text-[#6a6a8a] flex items-center gap-1'>
                  <Flame size={12} className='text-[#ff2a6d]' /> Trending
                  Suggestions:
                </span>
                <button
                  onClick={() => setTrendingExpanded(v => !v)}
                  className='text-[10px] text-[#05d9e8] hover:text-white font-bold hover:underline cursor-pointer'
                >
                  {trendingExpanded ? 'Collapse' : 'Expand'}
                </button>
              </div>

              {/* Category Pills */}
              <div className='flex items-center gap-1.5 overflow-x-auto pb-1 mb-3 scrollbar-hide'>
                {TRENDING_CATEGORIES.map(cat => (
                  <button
                    key={cat.id}
                    onClick={() => {
                      setSelectedTrendingCategory(cat);
                      setTrendingExpanded(false);
                    }}
                    style={selectedTrendingCategory?.id === cat.id ? {
                      backgroundImage: `linear-gradient(135deg, ${cat.color}, ${cat.color}aa)`,
                      borderColor: cat.color,
                      boxShadow: `0 0 10px ${cat.color}66`
                    } : undefined}
                    className={`px-3 py-1.5 rounded-full text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 border ${
                      selectedTrendingCategory?.id === cat.id
                        ? 'text-white'
                        : 'bg-[#05050a] border-[#1a1a2e] text-[#a0a0cf] hover:border-[#ff2a6d] hover:text-[#ff2a6d]'
                    }`}
                  >
                    <span className='text-[10px]'>{cat.emoji}</span>
                    {cat.label}
                  </button>
                ))}
              </div>

              {/* Trending Tags Grid */}
              {trendingExpanded && selectedTrendingCategory && (
                <div className='flex items-center gap-1.5 flex-wrap'>
                  {selectedTrendingCategory.tags.map(tag => (
                    <button
                      key={tag}
                      onClick={() => {
                        setQuery(tag);
                        handleExecute(1, tag);
                        setTrendingExpanded(false);
                      }}
                      className='px-2.5 py-1 bg-[#05050a] border border-[#1a1a2e] hover:border-[#ff2a6d] rounded text-[11px] text-[#e0e0ff] hover:text-[#ff2a6d] transition-all cursor-pointer'
                    >
                      #{tag}
                    </button>
                  ))}
                  {selectedTrendingCategory.id === 'all' && (
                    <>
                      {RANDOM_TAGS.map(tag => (
                        <button
                          key={tag}
                          onClick={() => {
                            const randomTag =
                              selectedTrendingCategory.tags[
                                Math.floor(
                                  Math.random() *
                                    selectedTrendingCategory.tags.length
                                )
                              ];
                            setQuery(randomTag);
                            handleExecute(1, randomTag);
                            setTrendingExpanded(false);
                          }}
                          className='px-2.5 py-1 bg-[#1a1a2e] border border-[#ff2a6d]/30 rounded text-[11px] text-[#ff2a6d] hover:bg-[#ff2a6d]/20 hover:text-white transition-all cursor-pointer'
                        >
                          ⚡ {tag}
                        </button>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Loading Indicator */}
          {loading && results.length === 0 && (
            <div className='flex flex-col items-center justify-center py-20 bg-[#0a0a12]/50 border border-[#1a1a2e] rounded-xl my-6'>
              <div className='w-14 h-14 border-4 border-[#1a1a2e] border-t-[#ff2a6d] border-r-[#05d9e8] rounded-full animate-spin shadow-[0_0_20px_rgba(255,42,109,0.3)]' />
              <p className='mt-5 text-[#ff2a6d] font-bold tracking-widest text-sm animate-pulse flex items-center gap-2'>
                <span>🔞</span> HUNTING {currentSiteConfig.name.toUpperCase()}
                ...
              </p>
              <div className='w-48 h-1 bg-gradient-to-r from-transparent via-[#ff2a6d] to-transparent mt-3 animate-loading-bar rounded' />
            </div>
          )}

          {/* Empty State */}
          {!loading && results.length === 0 && (
            <div className='text-center py-20 px-4 bg-[#0a0a12] border border-[#1a1a2e] rounded-xl my-4'>
              <Film
                size={44}
                className='mx-auto text-[#1a1a2e] mb-3 stroke-[1.5]'
              />
              <p className='text-lg font-semibold text-[#a0a0cf]'>
                {hasSearched ? 'No public listings found.' : 'Search public listings.'}
              </p>
              <p className='text-xs text-[#6a6a8a] mt-2 max-w-md mx-auto'>
                {hasSearched
                  ? 'Try a different phrase or source. Some sites may block automated requests or change their page layout.'
                  : `Choose a source and search its public listings. ${Object.keys(SITES).length} sources are configured.`}
              </p>
              <button
                onClick={() => setShowInstallModal(true)}
                className='mt-4 px-4 py-2 rounded-lg bg-[#05d9e8]/15 border border-[#05d9e8]/40 text-[#05d9e8] text-xs font-bold hover:bg-[#05d9e8]/25 transition-all inline-flex items-center gap-1.5 cursor-pointer'
              >
                <Smartphone size={14} />
                <span>Install Pyrmsearch on this device</span>
              </button>
            </div>
          )}

          {/* Results Grid */}
          {results.length > 0 && (
            <div>
              <div className='flex justify-between items-center mb-3 px-1'>
                <span className='text-xs text-[#6a6a8a]'>
                  Showing{' '}
                  <strong className='text-white'>{results.length}</strong>{' '}
                  videos from{' '}
                  <span className='text-[#05d9e8] font-semibold'>
                    {currentSiteConfig.name}
                  </span>
                  <span className='ml-2 text-[#05d9e8] text-[11px]'>
                    (Hover cards for video preview 🎬)
                  </span>
                </span>
                <span className='text-xs text-[#6a6a8a]'>Page {page}</span>
              </div>

              <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'>
                {results.map(item => (
                  <VideoCard
                    key={item.id}
                    item={item}
                    isFavorited={isFavorited(item.url)}
                    onToggleFavorite={handleToggleFavorite}
                    onCopyLink={handleCopyLink}
                    onOpenTeaser={v => setPreviewModalItem(v)}
                    onMarkWatched={addToRecentlyWatched}
                    onSaveScreenshot={handleSaveScreenshot}
                    onShare={handleShare}
                    collections={collections}
                    onAddToCollection={handleAddToCollection}
                  />
                ))}
              </div>

              {/* Load More & Infinite Scroll Status */}
              <div className='text-center py-8'>
                {loading ? (
                  <div className='flex flex-col items-center gap-2'>
                    <div className='w-8 h-8 border-3 border-[#1a1a2e] border-t-[#05d9e8] rounded-full animate-spin' />
                    <span className='text-xs text-[#05d9e8] font-semibold tracking-wider'>
                      LOADING PAGE {page + 1}...
                    </span>
                  </div>
                ) : hasMore ? (
                  <button
                    onClick={() => handleExecute(page + 1)}
                    className='px-8 py-3 rounded-lg border border-[#05d9e8] text-[#05d9e8] hover:bg-[#05d9e8] hover:text-black font-bold text-xs sm:text-sm tracking-widest transition-all shadow-[0_0_15px_rgba(5,217,232,0.3)] hover:shadow-[0_0_25px_rgba(5,217,232,0.6)] cursor-pointer'
                  >
                    LOAD MORE • PAGE {page + 1}
                  </button>
                ) : (
                  <div className='text-xs text-[#6a6a8a] py-3 tracking-wider uppercase'>
                    — End of results reached —
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: FAVORITES PANEL */}
      {activeTab === 'fav' && (
        <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-5 shadow-[0_0_15px_rgba(255,42,109,0.15)]'>
          <div className='flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 pb-4 border-b border-[#1a1a2e] mb-5'>
            <div>
              <h2 className='text-lg font-bold text-white flex items-center gap-2'>
                <span>🔞🔞</span> SAVED FAVORITES
              </h2>
              <p className='text-xs text-[#6a6a8a] mt-0.5'>
                Saved in local storage. Hover cards to play preview video or
                copy link.
              </p>
            </div>

            {favorites.length > 0 && (
              <div className='flex items-center gap-2'>
                <button
                  onClick={async () => {
                    const links = favorites.map(favorite => favorite.url).join('\n');
                    try {
                      await navigator.clipboard.writeText(links);
                      showToast(`Copied ${favorites.length} links.`);
                    } catch {
                      showToast('Clipboard access was denied.');
                    }
                  }}
                  className='px-3 py-1.5 text-xs font-semibold rounded bg-[#141424] border border-[#1a1a2e] hover:border-[#05d9e8] text-[#05d9e8] hover:bg-[#05d9e8]/10 transition-all flex items-center gap-1.5 cursor-pointer'
                >
                  <Copy size={13} /> COPY ALL LINKS
                </button>
                <button
                  onClick={handleClearFavorites}
                  className='px-3 py-1.5 text-xs font-semibold rounded border border-[#ff2a6d] text-[#ff2a6d] hover:bg-[#ff2a6d] hover:text-white transition-all flex items-center gap-1.5 cursor-pointer shadow-[0_0_8px_rgba(255,42,109,0.3)]'
                >
                  <Trash2 size={13} /> CLEAR ALL
                </button>
              </div>
            )}
          </div>

          {favorites.length === 0 ? (
            <div className='text-center py-20 px-4'>
              <Heart
                size={44}
                className='mx-auto text-[#1a1a2e] mb-3 stroke-[1.5]'
              />
              <p className='text-sm font-semibold text-[#a0a0cf]'>
                No favorites saved yet.
              </p>
              <p className='text-xs text-[#6a6a8a] mt-1.5'>
                Hunt videos and click the 🔞 icon on any card to save it here.
              </p>
              <button
                onClick={() => setActiveTab('hunt')}
                className='mt-4 px-5 py-2 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white text-xs font-bold tracking-wider hover:opacity-90 cursor-pointer shadow-[0_0_12px_rgba(255,42,109,0.4)]'
              >
                START HUNTING
              </button>
            </div>
          ) : (
            <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'>
              {favorites.map(item => (
                <VideoCard
                  key={item.id}
                  item={item}
                  isFavorited={isFavorited(item.url)}
                  onToggleFavorite={handleToggleFavorite}
                  onCopyLink={handleCopyLink}
                  onOpenTeaser={v => setPreviewModalItem(v)}
                  onMarkWatched={addToRecentlyWatched}
                  onSaveScreenshot={handleSaveScreenshot}
                  onShare={handleShare}
                  collections={collections}
                  onAddToCollection={handleAddToCollection}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 4: SAVED SCREENSHOTS */}
      {activeTab === 'screenshots' && (
        <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-5 shadow-[0_0_15px_rgba(255,42,109,0.15)]'>
          <div className='flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 pb-4 border-b border-[#1a1a2e] mb-5'>
            <div>
              <h2 className='text-lg font-bold text-white flex items-center gap-2'>
                <Camera size={18} className='text-[#05d9e8]' /> SAVED
                SCREENSHOTS
              </h2>
              <p className='text-xs text-[#6a6a8a] mt-0.5'>
                Captured video frames saved to local storage. Tap any screenshot
                to view full size.
              </p>
            </div>

            {screenshots.length > 0 && (
              <div className='flex items-center gap-2'>
                <button
                  onClick={async () => {
                    const dataUrls = screenshots.map(screenshot => screenshot.dataUrl).join('\n');
                    try {
                      await navigator.clipboard.writeText(dataUrls);
                      showToast(`Copied ${screenshots.length} screenshot data URLs.`);
                    } catch {
                      showToast('Could not copy the screenshots; use Backup to export them.');
                    }
                  }}
                  className='px-3 py-1.5 text-xs font-semibold rounded bg-[#141424] border border-[#1a1a2e] hover:border-[#05d9e8] text-[#05d9e8] hover:bg-[#05d9e8]/10 transition-all flex items-center gap-1.5 cursor-pointer'
                >
                  <Copy size={13} /> COPY ALL DATA URLs
                </button>
                <button
                  onClick={handleClearScreenshots}
                  className='px-3 py-1.5 text-xs font-semibold rounded border border-[#ff2a6d] text-[#ff2a6d] hover:bg-[#ff2a6d] hover:text-white transition-all flex items-center gap-1.5 cursor-pointer shadow-[0_0_8px_rgba(255,42,109,0.3)]'
                >
                  <Trash2 size={13} /> CLEAR ALL
                </button>
              </div>
            )}
          </div>

          {screenshots.length === 0 ? (
            <div className='text-center py-20 px-4'>
              <Camera
                size={44}
                className='mx-auto text-[#1a1a2e] mb-3 stroke-[1.5]'
              />
              <p className='text-sm font-semibold text-[#a0a0cf]'>
                No screenshots saved yet.
              </p>
              <p className='text-xs text-[#6a6a8a] mt-1.5'>
                Hunt videos, hover to play preview, then click the 📷 icon on
                any card to capture a frame.
              </p>
              <button
                onClick={() => setActiveTab('hunt')}
                className='mt-4 px-5 py-2 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white text-xs font-bold tracking-wider hover:opacity-90 cursor-pointer shadow-[0_0_12px_rgba(255,42,109,0.4)]'
              >
                START HUNTING
              </button>
            </div>
          ) : (
            <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'>
              {screenshots.map(shot => (
                <div
                  key={shot.id}
                  className='bg-[#05050a] border border-[#1a1a2e] rounded-xl overflow-hidden flex flex-col cursor-pointer group hover:border-[#05d9e8]/50 transition-all'
                >
                  <div className='relative w-full h-42 bg-black overflow-hidden'>
                    <a
                      href={shot.dataUrl}
                      target='_blank'
                      rel='noopener noreferrer'
                      aria-label={`Open full-size screenshot: ${shot.title}`}
                      className='absolute inset-0'
                    >
                      <img
                        src={shot.dataUrl}
                        alt={shot.title}
                        className='w-full h-full object-cover transition-transform duration-300 group-hover:scale-105'
                      />
                    </a>
                    <div
                      style={{ backgroundColor: shot.siteColor }}
                      className='absolute bottom-2 left-2 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-black/85 text-white border z-10 backdrop-blur-xs'
                    >
                      {shot.siteName}
                    </div>
                    <button
                      type='button'
                      aria-label={`Delete screenshot: ${shot.title}`}
                      onClick={e => {
                        e.stopPropagation();
                        handleDeleteScreenshot(shot.id);
                      }}
                      className='absolute top-2 right-2 bg-black/70 p-1.5 rounded text-white hover:bg-[#ff2a6d] transition-all cursor-pointer z-20'
                    >
                      <X size={12} />
                    </button>
                  </div>
                  <div className='p-2.5'>
                    <h3 className='text-xs font-semibold text-[#e0e0ff] line-clamp-2 mb-1'>
                      {shot.title}
                    </h3>
                    <div className='flex items-center justify-between text-[10px] text-[#6a6a8a]'>
                      <span className='font-mono'>
                        {new Date(shot.capturedAt).toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab: RECENTLY WATCHED */}
      {activeTab === 'recent' && (
        <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-5 shadow-[0_0_15px_rgba(255,42,109,0.15)]'>
          <div className='flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 pb-4 border-b border-[#1a1a2e] mb-5'>
            <div>
              <h2 className='text-lg font-bold text-white flex items-center gap-2'>
                <History size={18} className='text-[#05d9e8]' /> RECENTLY
                WATCHED
              </h2>
              <p className='text-xs text-[#6a6a8a] mt-0.5'>
                Videos opened in the teaser or sent to the source site. Tap to re-open.
              </p>
            </div>
            {recentlyWatched.length > 0 && (
              <button
                onClick={() => {
                  setRecentlyWatched([]);
                  try {
                    localStorage.removeItem('pyrmsearchRecentlyWatched');
                  } catch {
                    // The in-memory history is still cleared.
                  }
                  showToast('Recent history cleared');
                  addLog('Recent history cleared');
                }}
                className='px-3 py-1.5 text-xs font-semibold rounded border border-[#ff2a6d] text-[#ff2a6d] hover:bg-[#ff2a6d] hover:text-white transition-all flex items-center gap-1.5 cursor-pointer shadow-[0_0_8px_rgba(255,42,109,0.3)]'
              >
                <Trash2 size={13} /> CLEAR ALL
              </button>
            )}
          </div>

          {recentlyWatched.length === 0 ? (
            <div className='text-center py-20 px-4'>
              <History
                size={44}
                className='mx-auto text-[#1a1a2e] mb-3 stroke-[1.5]'
              />
              <p className='text-sm font-semibold text-[#a0a0cf]'>
                No recent history yet.
              </p>
              <p className='text-xs text-[#6a6a8a] mt-1.5'>
                Open a teaser or use WATCH to add a listing here.
              </p>
              <button
                onClick={() => setActiveTab('hunt')}
                className='mt-4 px-5 py-2 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white text-xs font-bold tracking-wider hover:opacity-90 cursor-pointer shadow-[0_0_12px_rgba(255,42,109,0.4)]'
              >
                START HUNTING
              </button>
            </div>
          ) : (
            <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'>
              {recentlyWatched.map(item => (
                <div
                  key={item.id}
                  role='button'
                  tabIndex={0}
                  aria-label={`Reopen ${item.title}`}
                  className='bg-[#05050a] border border-[#1a1a2e] rounded-xl overflow-hidden flex flex-col cursor-pointer group hover:border-[#05d9e8]/50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#05d9e8] transition-all'
                  onClick={() => setPreviewModalItem(item)}
                  onKeyDown={event => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setPreviewModalItem(item);
                    }
                  }}
                >
                  <div className='relative w-full h-42 bg-black overflow-hidden'>
                    {item.thumb && (
                      <img
                        src={item.thumb}
                        alt={item.title}
                        className='w-full h-full object-cover transition-transform duration-300 group-hover:scale-105'
                      />
                    )}
                    <div
                      style={{ backgroundColor: item.siteColor }}
                      className='absolute bottom-2 left-2 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-black/85 text-white border z-10 backdrop-blur-xs'
                    >
                      {item.siteName}
                    </div>
                  </div>
                  <div className='p-2.5'>
                    <h3 className='text-xs font-semibold text-[#e0e0ff] line-clamp-2 mb-1'>
                      {item.title}
                    </h3>
                    <div className='flex items-center justify-between text-[10px] text-[#6a6a8a]'>
                      <span className='font-mono'>
                        {new Date(item.watchedAt).toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab: COLLECTIONS */}
      {activeTab === 'collections' && (
        <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-5 shadow-[0_0_15px_rgba(255,42,109,0.15)]'>
          <div className='flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 pb-4 border-b border-[#1a1a2e] mb-5'>
            <div>
              <h2 className='text-lg font-bold text-white flex items-center gap-2'>
                <SaveIcon size={18} className='text-[#05d9e8]' /> COLLECTIONS
              </h2>
              <p className='text-xs text-[#6a6a8a] mt-0.5'>
                Organize saved videos into named playlists.
              </p>
            </div>
            {collections.length > 0 && (
              <button
                onClick={handleClearCollections}
                className='px-3 py-1.5 text-xs font-semibold rounded border border-[#ff2a6d] text-[#ff2a6d] hover:bg-[#ff2a6d] hover:text-white transition-all flex items-center gap-1.5 cursor-pointer shadow-[0_0_8px_rgba(255,42,109,0.3)]'
              >
                <Trash2 size={13} /> CLEAR ALL
              </button>
            )}
          </div>

          {/* Create New Collection */}
          <div className='bg-[#05050a] border border-[#1a1a2e] rounded-xl p-4 mb-6'>
            <h3 className='text-sm font-bold text-white mb-3'>
              Create New Collection
            </h3>
            <div className='flex flex-col sm:flex-row gap-3'>
              <input
                value={newCollectionName}
                maxLength={80}
                onChange={e => setNewCollectionName(e.target.value)}
                placeholder='Collection name (e.g. Blondes, 4K, Amateur)'
                className='flex-1 bg-[#0a0a12] border border-[#1a1a2e] text-white rounded-lg px-3 py-2.5 text-sm outline-none focus:border-[#05d9e8]'
              />
              <input
                type='color'
                value={newCollectionColor}
                onChange={e => setNewCollectionColor(e.target.value)}
                className='w-10 h-10 rounded-lg border-none bg-transparent cursor-pointer'
              />
              <button
                onClick={handleCreateCollection}
                className='px-4 py-2.5 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white text-xs font-bold tracking-wider hover:opacity-95 cursor-pointer shadow-[0_0_12px_rgba(255,42,109,0.4)]'
              >
                CREATE
              </button>
            </div>
          </div>

          {collections.length === 0 ? (
            <div className='text-center py-20 px-4'>
              <SaveIcon
                size={44}
                className='mx-auto text-[#1a1a2e] mb-3 stroke-[1.5]'
              />
              <p className='text-sm font-semibold text-[#a0a0cf]'>
                No collections yet.
              </p>
              <p className='text-xs text-[#6a6a8a] mt-1.5'>
                Create a collection above, then add videos from Hunt or
                Favorites.
              </p>
            </div>
          ) : (
            <div className='space-y-4'>
              {collections.map(collection => (
                <div
                  key={collection.id}
                  className='bg-[#05050a] border border-[#1a1a2e] rounded-xl p-4 overflow-hidden'
                >
                  <div className='flex items-start justify-between mb-3'>
                    <div className='flex items-center gap-2'>
                      <span
                        style={{ backgroundColor: collection.color }}
                        className='px-2 py-0.5 rounded text-[11px] font-black text-white'
                      >
                        {collection.color}
                      </span>
                      <h4 className='font-bold text-sm text-white'>
                        {collection.name}
                      </h4>
                    </div>
                    <button
                      type='button'
                      aria-label={`Delete collection: ${collection.name}`}
                      onClick={() => handleDeleteCollection(collection.id)}
                      className='text-[#6a6a8a] hover:text-[#ff2a6d] p-1 rounded hover:bg-[#ff2a6d]/20 cursor-pointer'
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <div className='flex items-center justify-between text-xs text-[#6a6a8a] mb-3'>
                    <span>{collection.videoIds.length} videos</span>
                    <span className='font-mono'>
                      {new Date(collection.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  {collection.videoIds.length === 0 ? (
                    <div className='text-center py-4 text-xs text-[#6a6a8a] bg-[#0a0a12]/50 rounded-lg border border-[#1a1a2e]/50'>
                      Empty — add videos from Hunt or Favorites
                    </div>
                  ) : (
                    <div className='grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3'>
                      {collection.videoIds.map(url => {
                        const item =
                          collection.videos.find(video => video.url === url) ||
                          favorites.find(f => f.url === url) ||
                          recentlyWatched.find(r => r.url === url);
                        if (!item) return null;
                        return (
                          <div
                            key={url}
                            role='button'
                            tabIndex={0}
                            aria-label={`Open ${item.title} from ${collection.name}`}
                            className='bg-[#0a0a12] border border-[#1a1a2e] rounded-lg p-2 cursor-pointer hover:border-[#05d9e8]/50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#05d9e8] transition-all group'
                            onClick={event => {
                              event.stopPropagation();
                              setPreviewModalItem(item as VideoResult);
                            }}
                            onKeyDown={event => {
                              if (event.target !== event.currentTarget) return;
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault();
                                setPreviewModalItem(item as VideoResult);
                              }
                            }}
                          >
                            <div className='relative w-full h-24 bg-black overflow-hidden mb-2'>
                              {item.thumb && (
                                <img
                                  src={item.thumb}
                                  alt={item.title}
                                  className='w-full h-full object-cover group-hover:scale-105 transition-transform'
                                />
                              )}
                              <button
                                type='button'
                                aria-label={`Remove ${item.title} from ${collection.name}`}
                                onClick={e => {
                                  e.stopPropagation();
                                  handleRemoveFromCollection(
                                    collection.id,
                                    url
                                  );
                                }}
                                className='absolute top-1 right-1 bg-black/70 p-1 rounded text-white hover:bg-[#ff2a6d] cursor-pointer z-10'
                              >
                                <X size={10} />
                              </button>
                            </div>
                            <h5 className='text-[10px] font-semibold text-[#e0e0ff] line-clamp-1'>
                              {item.title}
                            </h5>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab: IMPORT / EXPORT */}
      {activeTab === 'import-export' && (
        <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-5 shadow-[0_0_15px_rgba(255,42,109,0.15)]'>
          <div className='flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 pb-4 border-b border-[#1a1a2e] mb-6'>
            <div>
              <h2 className='text-lg font-bold text-white flex items-center gap-2'>
                <Upload size={18} className='text-[#05d9e8]' /> DATA BACKUP
              </h2>
              <p className='text-xs text-[#6a6a8a] mt-0.5'>
                Export your favorites, screenshots, history, and collections to
                a JSON file. Import to restore on another device.
              </p>
            </div>
          </div>

          <div className='grid grid-cols-1 sm:grid-cols-2 gap-6'>
            {/* Export Card */}
            <div className='bg-[#05050a] border border-[#1a1a2e] rounded-xl p-5'>
              <h3 className='text-sm font-bold text-white mb-4 flex items-center gap-2'>
                <Download size={16} className='text-[#05d9e8]' /> Export Backup
              </h3>
              <p className='text-xs text-[#6a6a8a] mb-5'>
                Downloads a JSON file containing all your saved data:
              </p>
              <ul className='text-xs text-[#a0a0cf] space-y-1 mb-6'>
                <li>• Favorites ({favorites.length})</li>
                <li>• Screenshots ({screenshots.length})</li>
                <li>• Search History ({history.length})</li>
                <li>• Recently Watched ({recentlyWatched.length})</li>
                <li>• Collections ({collections.length})</li>
              </ul>
              <button
                onClick={handleExportData}
                disabled={isExporting}
                className='w-full py-3 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] text-white font-bold text-xs tracking-wider hover:opacity-95 cursor-pointer shadow-[0_0_15px_rgba(255,42,109,0.4)] disabled:opacity-50 transition-all flex items-center justify-center gap-2'
              >
                {isExporting ? (
                  'EXPORTING...'
                ) : (
                  <>
                    <Download size={14} /> <span>EXPORT JSON</span>
                  </>
                )}
              </button>
            </div>

            {/* Import Card */}
            <div className='bg-[#05050a] border border-[#1a1a2e] rounded-xl p-5'>
              <h3 className='text-sm font-bold text-white mb-4 flex items-center gap-2'>
                <SaveIcon size={16} className='text-[#05d9e8]' /> Import Backup
              </h3>
              <p className='text-xs text-[#6a6a8a] mb-5'>
                Select a previously exported pyrmsearch JSON backup file to
                restore your data.
              </p>
              <label className='block cursor-pointer'>
                <input
                  type='file'
                  accept='.json'
                  onChange={e => {
                    if (e.target.files?.[0]) {
                      setImportFile(e.target.files[0]);
                      handleImportData(e.target.files[0]);
                    }
                  }}
                  className='sr-only'
                />
                <div
                  className={`flex flex-col items-center justify-center gap-2 p-4 rounded-lg border-2 border-dashed transition-all ${isImporting ? 'border-[#ff2a6d] bg-[#ff2a6d]/10' : 'border-[#1a1a2e] hover:border-[#05d9e8] bg-[#0a0a12]'}`}
                >
                  <Upload
                    size={24}
                    className={
                      isImporting ? 'text-[#ff2a6d]' : 'text-[#6a6a8a]'
                    }
                  />
                  <span
                    className={`text-xs font-semibold ${isImporting ? 'text-[#ff2a6d]' : 'text-[#a0a0cf]'}`}
                  >
                    {isImporting ? 'IMPORTING...' : 'Choose JSON file'}
                  </span>
                </div>
              </label>
              {importFile && (
                <div className='mt-3 p-3 rounded-lg bg-[#0a0a12]/50 border border-[#1a1a2e]/50'>
                  <p className='text-xs text-[#a0a0cf] font-mono truncate'>
                    {importFile.name}
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className='mt-8 pt-6 border-t border-[#1a1a2e]'>
            <h3 className='text-sm font-bold text-white mb-3'>Storage Info</h3>
            <div className='text-xs text-[#6a6a8a] space-y-1'>
              <p>
                All data is stored in browser <code>localStorage</code>.
              </p>
              <p>
                Clearing browser data will delete your saved content unless you
                export a backup first.
              </p>
              <p>
                Backups are portable — import them on any device running
                pyrmsearch.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: SITE DIAGNOSTICS & TEST ALL SITES */}
      {activeTab === 'test' && (
        <div className='bg-[#0a0a12] border border-[#1a1a2e] rounded-xl p-4 sm:p-5 shadow-[0_0_15px_rgba(5,217,232,0.15)]'>
          <div className='flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 pb-4 border-b border-[#1a1a2e] mb-4'>
            <div>
              <h2 className='text-lg font-bold text-white flex items-center gap-2'>
                <Activity size={18} className='text-[#05d9e8]' /> SITE
                CONNECTIVITY & HEALTH DIAGNOSTICS
              </h2>
              <p className='text-xs text-[#6a6a8a] mt-0.5'>
                Real-time ping, HTTP responses, and latency for all{' '}
                {Object.keys(SITES).length} supported adult tube sites.
              </p>
            </div>

            <button
              onClick={handleRunSiteTests}
              disabled={testingSites}
              className='flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-[#05d9e8] to-[#00b4d8] text-black font-extrabold text-xs tracking-wider cursor-pointer shadow-[0_0_15px_rgba(5,217,232,0.4)] hover:shadow-[0_0_25px_rgba(5,217,232,0.7)] disabled:opacity-50 transition-all'
            >
              <RotateCcw
                size={14}
                className={testingSites ? 'animate-spin' : ''}
              />
              <span>
                {testingSites ? 'TESTING ALL SITES...' : 'RE-TEST ALL 13 SITES'}
              </span>
            </button>
          </div>

          {/* Test Status Bar & Filter Controls */}
          <div className='flex flex-wrap items-center justify-between gap-3 bg-[#070710] border border-[#1a1a2e] rounded-xl p-3 mb-4'>
            <div className='flex items-center gap-2 sm:gap-4 text-xs'>
              <span className='text-[#6a6a8a]'>
                Operational:{' '}
                <strong className='text-emerald-400'>{onlineSiteCount}</strong>{' '}
                / {Object.keys(SITES).length}
              </span>
              <span className='text-[#6a6a8a] hidden sm:inline'>•</span>
              <span className='text-[#6a6a8a] hidden sm:inline'>
                Hover Video Previews:{' '}
                <strong className='text-[#05d9e8]'>Supported</strong>
              </span>
            </div>

            <div className='flex items-center gap-1.5'>
              <button
                onClick={() => setTestFilter('all')}
                className={`px-3 py-1 rounded text-xs font-semibold cursor-pointer transition-all ${
                  testFilter === 'all'
                    ? 'bg-[#05d9e8] text-black font-bold'
                    : 'bg-[#141424] text-[#6a6a8a] hover:text-white'
                }`}
              >
                All ({Object.keys(SITES).length})
              </button>
              <button
                onClick={() => setTestFilter('online')}
                className={`px-3 py-1 rounded text-xs font-semibold cursor-pointer transition-all ${
                  testFilter === 'online'
                    ? 'bg-emerald-500 text-black font-bold'
                    : 'bg-[#141424] text-[#6a6a8a] hover:text-white'
                }`}
              >
                Online ({onlineSiteCount})
              </button>
              <button
                onClick={() => setTestFilter('error')}
                className={`px-3 py-1 rounded text-xs font-semibold cursor-pointer transition-all ${
                  testFilter === 'error'
                    ? 'bg-amber-500 text-black font-bold'
                    : 'bg-[#141424] text-[#6a6a8a] hover:text-white'
                }`}
              >
                Errors ({siteTests.filter(test => !test.ok).length})
              </button>
            </div>
          </div>

          {/* Site Cards Grid */}
          <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5'>
            {Object.values(SITES)
              .filter(cfg => {
                const test = siteTests.find(t => t.id === cfg.id);
                if (testFilter === 'online') return test?.ok;
                if (testFilter === 'error') return test && !test.ok;
                return true;
              })
              .map(cfg => {
                const test = siteTests.find(t => t.id === cfg.id);
                const isTested = Boolean(test);
                const isOk = test?.ok;

                return (
                  <div
                    key={cfg.id}
                    className='bg-[#05050a] border border-[#1a1a2e] rounded-xl p-4 flex flex-col justify-between hover:border-[#05d9e8]/50 transition-all shadow-sm'
                  >
                    <div>
                      <div className='flex items-start justify-between'>
                        <div>
                          <div className='flex items-center gap-2'>
                            <span
                              style={{ backgroundColor: cfg.color }}
                              className='px-2 py-0.5 rounded text-[11px] font-black text-white'
                            >
                              {cfg.icon}
                            </span>
                            <h4 className='font-bold text-sm text-white'>
                              {cfg.name}
                            </h4>
                          </div>
                          <a
                            href={cfg.base}
                            target='_blank'
                            rel='noopener noreferrer'
                            className='text-xs text-[#6a6a8a] hover:text-[#05d9e8] mt-1 inline-block font-mono'
                          >
                            {cfg.base}
                          </a>
                        </div>

                        {/* Status Badge */}
                        <div>
                          {testingSites && !test ? (
                            <span className='px-2.5 py-1 rounded text-[11px] bg-[#141424] text-[#a0a0cf] animate-pulse'>
                              Testing...
                            </span>
                          ) : isTested ? (
                            isOk ? (
                              <span className='flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-bold bg-emerald-950/80 border border-emerald-500/50 text-emerald-400'>
                                <CheckCircle2 size={13} /> {test.status} OK (
                                {test.latencyMs}ms)
                              </span>
                            ) : (
                              <span className='flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-bold bg-amber-950/80 border border-amber-500/50 text-amber-400'>
                                <AlertCircle size={13} />{' '}
                                {test.status || 'ERROR'} ({test.latencyMs}ms)
                              </span>
                            )
                          ) : (
                            <span className='px-2 py-0.5 rounded text-[11px] bg-[#141424] text-[#6a6a8a]'>
                              Not tested
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Site Features Badge */}
                      <div className='flex items-center gap-2 mt-3 text-[11px] text-[#6a6a8a]'>
                        <span className='px-2 py-0.5 rounded bg-[#121224] text-[#a0a0cf]'>
                          Pagination: <code>{cfg.paginate}</code>
                        </span>
                      </div>
                    </div>

                    <div className='flex items-center justify-between mt-4 pt-3 border-t border-[#1a1a2e]'>
                      <button
                        onClick={() => handleTestSingleSite(cfg.id)}
                        className='px-2.5 py-1 rounded text-xs text-[#a0a0cf] hover:text-white bg-[#141424] hover:bg-[#1e1e36] border border-[#1a1a2e] transition-all cursor-pointer flex items-center gap-1'
                      >
                        <RotateCcw size={11} /> Ping
                      </button>

                      <button
                        onClick={() => {
                          setSite(cfg.id);
                          setActiveTab('hunt');
                          showToast(`Switched active search to ${cfg.name}`);
                          addLog(`Selected site: ${cfg.name}`);
                          if (query) {
                            handleExecute(1, query, cfg.id);
                          }
                        }}
                        className='px-3 py-1 rounded text-xs font-semibold bg-[#121224] hover:bg-[#ff2a6d]/20 hover:text-[#ff2a6d] border border-[#1a1a2e] text-white transition-all cursor-pointer'
                      >
                        SELECT & HUNT
                      </button>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Floating Cyberpunk Real-Time Console */}
      <div className='fixed bottom-0 left-0 right-0 z-40 bg-[#05050c]/95 border-t border-[#1a1a2e] backdrop-blur-md'>
        <div className='flex items-center justify-between px-4 py-1.5 bg-[#0a0a14] border-b border-[#1a1a2e]/60 text-xs text-[#05d9e8] font-mono'>
          <div className='flex items-center gap-2'>
            <span className='inline-block w-2 h-2 rounded-full bg-[#05d9e8] animate-pulse' />
            <span className='font-bold'>PYRMSEARCH SYSTEM CONSOLE</span>
          </div>
          <div className='flex items-center gap-2'>
            <button
              onClick={() => setLogs([])}
              className='text-[10px] text-[#6a6a8a] hover:text-[#ff2a6d] px-1 cursor-pointer'
            >
              CLEAR
            </button>
            <button
              onClick={() => setConsoleOpen(v => !v)}
              className='text-[#6a6a8a] hover:text-white cursor-pointer'
            >
              {consoleOpen ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronUp size={14} />
              )}
            </button>
          </div>
        </div>

        {consoleOpen && (
          <div className='p-2 sm:p-3 font-mono text-[11px] text-[#05d9e8] max-h-28 overflow-y-auto space-y-1'>
            {logs.map((log, i) => (
              <div key={i} className='opacity-90 leading-tight'>
                {log}
              </div>
            ))}
            <div ref={consoleBottomRef} />
          </div>
        )}
      </div>
    </div>
  );
}
