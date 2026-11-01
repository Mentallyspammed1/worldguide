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
  ArrowDownToLine,
  ShieldCheck,
  Zap,
  Camera,
  Save,
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

interface VideoCardProps {
  item: VideoResult;
  isFavorited: boolean;
  onToggleFavorite: (e: React.MouseEvent, item: VideoResult) => void;
  onCopyLink: (e: React.MouseEvent, item: VideoResult) => void;
  onOpenTeaser: (item: VideoResult) => void;
  onSaveScreenshot: (item: VideoResult) => void;
  onShare: (item: VideoResult) => void;
}

// Self-contained Video Card component to isolate hover, preview, image error, and copy states
const VideoCard: React.FC<VideoCardProps> = ({
  item,
  isFavorited,
  onToggleFavorite,
  onCopyLink,
  onOpenTeaser,
  onSaveScreenshot,
  onShare
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isTouchActive, setIsTouchActive] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoSrc, setVideoSrc] = useState<string>('');
  const [isRetryingProxy, setIsRetryingProxy] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [isCopied, setIsCopied] = useState(false);
  const [isPipActive, setIsPipActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const preloadTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef(true);
  const modalAbortRef = useRef<AbortController | null>(null);

  const hasVideo = Boolean(videoSrc) && !videoError;
  const shouldShowVideo = (isHovered || isTouchActive) && hasVideo;

  // Track mounted state to prevent state updates after unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (preloadTimeoutRef.current) clearTimeout(preloadTimeoutRef.current);
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.src = '';
        videoRef.current.load();
      }
    };
  }, []);

  // Reset all media and hover states when item changes (e.g. after tests or new search)
  useEffect(() => {
    setIsHovered(false);
    setIsTouchActive(false);
    setImgError(false);
    setVideoError(false);
    setVideoLoaded(false);
    setVideoLoading(false);
    setIsRetryingProxy(false);
    setRetryCount(0);
    setVideoSrc('');
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = '';
      videoRef.current.load();
    }
  }, [item.id, item.previewVideo]);

  // Preload video metadata when card enters viewport (Intersection Observer)
  useEffect(() => {
    if (!item.previewVideo) return;

    const observer = new IntersectionObserver(
      entries => {
        const [entry] = entries;
        if (
          entry?.isIntersecting &&
          !videoSrc &&
          !videoError &&
          !videoLoading
        ) {
          // Delay preload slightly to avoid flooding on rapid scroll
          preloadTimeoutRef.current = setTimeout(() => {
            if (isMountedRef.current && !videoSrc && !videoError) {
              setVideoSrc(item.previewVideo);
              setVideoLoading(true);
            }
          }, 150);
        }
      },
      { threshold: 0.1, rootMargin: '100px' }
    );

    const el = containerRef.current;
    if (el) observer.observe(el);

    return () => {
      if (preloadTimeoutRef.current) clearTimeout(preloadTimeoutRef.current);
      if (modalAbortRef.current) modalAbortRef.current.abort();
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.src = '';
        videoRef.current.load();
      }
      // disconnect the IntersectionObserver that was set up below
      const obs = containerRef.current?.observer;
      if (obs) obs.disconnect();
    };
  }, [item.previewVideo, item.id]);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    onCopyLink(e, item);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleVideoLoadedData = () => {
    setVideoLoaded(true);
    setVideoLoading(false);
    // Auto-play when loaded and visible
    if (shouldShowVideo && videoRef.current) {
      videoRef.current.play().catch(() => {
        // Autoplay blocked, user interaction needed
      });
    }
  };

  const handleVideoError = () => {
    setVideoLoading(false);
    // If direct CDN video fails, attempt proxy fallback via backend (max 2 retries)
    if (
      retryCount < 2 &&
      item.previewVideo &&
      !item.previewVideo.startsWith('/api/proxy-media') &&
      !videoError
    ) {
      setRetryCount(prev => prev + 1);
      setIsRetryingProxy(true);
      setVideoSrc(
        `/api/proxy-media?url=${encodeURIComponent(item.previewVideo)}`
      );
    } else {
      setVideoError(true);
      setVideoLoaded(false);
    }
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
      onClick={() => window.open(item.url, '_blank', 'noopener,noreferrer')}
      className='group bg-[#0a0a12] border border-[#1a1a2e] hover:border-[#ff2a6d] rounded-xl overflow-hidden relative transition-all duration-300 hover:-translate-y-1.5 hover:shadow-[0_8px_25px_rgba(255,42,109,0.3)] flex flex-col cursor-pointer'
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
        {shouldShowVideo && (
          <video
            ref={videoRef}
            src={videoSrc}
            autoPlay
            muted
            playsInline
            loop
            preload='metadata'
            referrerPolicy='no-referrer'
            onLoadedData={handleVideoLoadedData}
            onCanPlay={() => setVideoLoaded(true)}
            onError={handleVideoError}
            className={`absolute inset-0 w-full h-full object-cover z-10 transition-opacity duration-300 ${
              videoLoaded ? 'opacity-100' : 'opacity-0'
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
          document.pictureInPictureEnabled && (
          <button
            onClick={e => {
              e.stopPropagation();
              if (document.pictureInPictureElement) {
                document.exitPictureInPicture().catch(() => {});
                setIsPipActive(false);
              } else {
                  videoRef.current!.requestPictureInPicture().catch(() => {});
                  setIsPipActive(true);
              }
            }}
            title={
              isPipActive ? 'Exit Picture-in-Picture' : 'Picture-in-Picture'
            }
            className='absolute top-2.5 right-11 w-8 h-8 rounded-full border border-[#05d9e8]/50 flex items-center justify-center text-xs transition-all z-20 cursor-pointer hover:bg-[#05d9e8]/20 hover:text-[#05d9e8] hover:border-[#05d9e8]'
          >
            <PictureInPicture2 size={14} />
          </button>
        )}

        {/* Favorite Toggle Button on Top-Right */}
        <button
          onClick={e => onToggleFavorite(e, item)}
          title={isFavorited ? 'Remove from favorites' : 'Save to favorites'}
          className={`absolute top-2.5 right-2.5 w-8 h-8 rounded-full border border-black/40 flex items-center justify-center text-sm transition-all z-20 cursor-pointer ${isFavorited ? 'bg-[#ff2a6d] text-white shadow-[0_0_12px_#ff2a6d] animate-pulse-neon' : 'bg-black/75 text-white/80 hover:bg-[#ff2a6d] hover:text-white hover:scale-110'}`}
        >
          {isFavorited ? '🔞🔞' : '🔞'}
        </button>
        <button
          onClick={e => {
            e.stopPropagation();
            const cardEl = document.querySelector(
              `[data-card-id="${item.id}"]`
            );
            const videoEl = cardEl?.querySelector('video');
            const dataUrl = captureVideoFrame(videoEl || null);
            if (dataUrl) handleSaveScreenshot(item, dataUrl);
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

          {/* Share Button */}
          <button
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
            onClick={e => e.stopPropagation()}
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
  const [activeTab, setActiveTab] = useState<'hunt' | 'fav' | 'test'>('hunt');
  const [query, setQuery] = useState('');
  const [site, setSite] = useState<string>('xvideos');
  const [page, setPage] = useState(1);
  const [results, setResults] = useState<VideoResult[]>([]);
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
  const [modalMuted, setModalMuted] = useState(false);
  const [modalVideoSrc, setModalVideoSrc] = useState<string>('');
  const [modalFullVideoSrc, setModalFullVideoSrc] = useState<string>('');
  const [modalUsingFull, setModalUsingFull] = useState(false);
  const [modalVideoLoaded, setModalVideoLoaded] = useState(false);
  const [modalVideoRetried, setModalVideoRetried] = useState(false);
  const [modalFetchingFull, setModalFetchingFull] = useState(false);

  // Sync modal video source when previewModalItem opens or changes
  // Also fetch the full video URL from the video page so users can play the entire video
  useEffect(() => {
    let cancelled = false;

    async function initModal() {
      if (!previewModalItem) {
        setModalVideoSrc('');
        setModalFullVideoSrc('');
        setModalUsingFull(false);
        setModalVideoLoaded(false);
        setModalVideoRetried(false);
        setModalFetchingFull(false);
        return;
      }

      // Add to recently watched when modal opens
      addToRecentlyWatched(previewModalItem);

      setModalVideoLoaded(false);
      setModalVideoRetried(false);
      setModalFetchingFull(true);

      // Start with preview video if available
      if (previewModalItem.previewVideo) {
        setModalVideoSrc(previewModalItem.previewVideo);
        setModalUsingFull(false);
      } else {
        setModalVideoSrc('');
        setModalUsingFull(false);
      }

      // Fetch full video URL from the video page
      try {
        const fullUrl = await fetchFullVideoUrl(previewModalItem.url);
        if (!cancelled && fullUrl) {
          setModalFullVideoSrc(fullUrl);
        }
      } catch {
        // ignore
      } finally {
        if (!cancelled) setModalFetchingFull(false);
      }
    }

    initModal();
    return () => {
      cancelled = true;
    };
  }, [previewModalItem, addToRecentlyWatched]);

  const handleModalVideoError = useCallback(() => {
    if (
      !modalVideoRetried &&
      previewModalItem?.previewVideo &&
      !modalVideoSrc.startsWith('/api/proxy-media')
    ) {
      setModalVideoRetried(true);
      setModalVideoSrc(
        `/api/proxy-media?url=${encodeURIComponent(previewModalItem.previewVideo)}`
      );
    }
  }, [modalVideoRetried, previewModalItem, modalVideoSrc]);

  // Download / Install APK Modal state
  const [showApkModal, setShowApkModal] = useState(false);

  // PWA & Android WebAPK install hook
  const { isInstallable, isInstalled, isIOS, isAndroid, install } =
    usePWAInstall();

  // Favorites state
  const [favorites, setFavorites] = useState<VideoResult[]>([]);

  // Saved video screenshots state
  interface SavedScreenshot {
    id: string;
    title: string;
    siteName: string;
    siteColor: string;
    thumb: string;
    dataUrl: string; // base64 PNG of captured video frame
    capturedAt: number;
  }
  const [screenshots, setScreenshots] = useState<SavedScreenshot[]>([]);

  // Search history state
  const [history, setHistory] = useState<string[]>([]);

  // Recently watched history state
  interface RecentlyWatchedItem {
    id: string;
    title: string;
    siteName: string;
    siteColor: string;
    thumb: string;
    url: string;
    previewVideo?: string;
    duration?: string;
    watchedAt: number;
  }
  const [recentlyWatched, setRecentlyWatched] = useState<RecentlyWatchedItem[]>(
    []
  );

  // Collections / Playlists state
  interface Collection {
    id: string;
    name: string;
    color: string;
    videoIds: string[]; // URLs of videos in this collection
    createdAt: number;
  }
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

  // Track seen URLs to avoid duplicate video cards in pagination
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

  // Load favorites & search history on mount
  useEffect(() => {
    try {
      const savedFavs = localStorage.getItem('pyrmsearchFavorites');
      if (savedFavs) {
        setFavorites(JSON.parse(savedFavs));
      }
    } catch {
      setFavorites([]);
    }

    try {
      const savedScreenshots = localStorage.getItem('pyrmsearchScreenshots');
      if (savedScreenshots) {
        setScreenshots(JSON.parse(savedScreenshots));
      }
    } catch {
      setScreenshots([]);
    }

    try {
      const savedHist = localStorage.getItem('searchHistory');
      if (savedHist) {
        setHistory(JSON.parse(savedHist));
      }
    } catch {
      setHistory([]);
    }

    try {
      const savedRecent = localStorage.getItem('pyrmsearchRecentlyWatched');
      if (savedRecent) {
        setRecentlyWatched(JSON.parse(savedRecent));
      }
    } catch {
      setRecentlyWatched([]);
    }

    try {
      const savedCollections = localStorage.getItem('pyrmsearchCollections');
      if (savedCollections) {
        setCollections(JSON.parse(savedCollections));
      }
    } catch {
      setCollections([]);
    }

    addLog(
      '🔞 pyrmsearch online • 13 Sites armed • Hover-Video-Preview active • APK ready'
    );
  }, [addLog]);

  // Scroll console to bottom on new log
  useEffect(() => {
    if (consoleOpen && consoleBottomRef.current) {
      consoleBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, consoleOpen]);

  // Copy link to clipboard with bulletproof iframe-safe fallback
  const handleCopyLink = useCallback(
    async (e: React.MouseEvent, item: VideoResult) => {
      e.stopPropagation();
      let copied = false;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(item.url);
          copied = true;
        }
      } catch {
        // ignore and try fallback
      }

      if (!copied) {
        try {
          const input = document.createElement('textarea');
          input.value = item.url;
          input.setAttribute('readonly', '');
          input.style.position = 'absolute';
          input.style.left = '-9999px';
          document.body.appendChild(input);
          input.select();
          document.execCommand('copy');
          document.body.removeChild(input);
          copied = true;
        } catch (err) {
          console.error('Copy fallback failed', err);
        }
      }

      showToast('Link copied to clipboard!');
      addLog(`📋 Copied: ${item.title.substring(0, 36)}...`);
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
        }
        return updated;
      });
    },
    [addLog, showToast]
  );

  // Capture a video frame as a base64 PNG screenshot
  const captureVideoFrame = useCallback(
    (videoEl: HTMLVideoElement | null): string | null => {
      if (!videoEl || videoEl.videoWidth === 0 || videoEl.videoHeight === 0)
        return null;
      const canvas = document.createElement('canvas');
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      ctx.drawImage(videoEl, 0, 0);
      return canvas.toDataURL('image/png');
    },
    []
  );

  // Save a screenshot to localStorage
  const handleSaveScreenshot = useCallback(
    (item: VideoResult, dataUrl: string) => {
      setScreenshots(prev => {
        const exists = prev.some(s => s.id === item.id);
        if (exists) {
          showToast('Screenshot already saved for this video');
          return prev;
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
        const updated = [entry, ...prev];
        try {
          localStorage.setItem(
            'pyrmsearchScreenshots',
            JSON.stringify(updated)
          );
        } catch (err) {
          console.error('Screenshot storage quota error', err);
          showToast('Storage full — could not save screenshot');
          return prev;
        }
        addLog(`📸 Screenshot saved: ${item.title.substring(0, 36)}...`);
        showToast('Screenshot saved!');
        return updated;
      });
    },
    [addLog, showToast]
  );

  // Delete a saved screenshot
  const handleDeleteScreenshot = useCallback(
    (id: string) => {
      setScreenshots(prev => {
        const updated = prev.filter(s => s.id !== id);
        try {
          localStorage.setItem(
            'pyrmsearchScreenshots',
            JSON.stringify(updated)
          );
        } catch {
          // ignore
        }
        addLog('Screenshot deleted');
        showToast('Screenshot deleted');
        return updated;
      });
    },
    [addLog, showToast]
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

  // Add item to recently watched history
  const addToRecentlyWatched = useCallback((item: VideoResult) => {
    setRecentlyWatched(prev => {
      const exists = prev.some(r => r.url === item.url);
      if (exists) {
        // Move to front
        const updated = [item, ...prev.filter(r => r.url !== item.url)];
        try {
          localStorage.setItem(
            'pyrmsearchRecentlyWatched',
            JSON.stringify(updated)
          );
        } catch {
          // ignore
        }
        return updated.slice(0, 50);
      }
      const entry: RecentlyWatchedItem = {
        id: item.id,
        title: item.title,
        siteName: item.siteName,
        siteColor: item.siteColor,
        thumb: item.thumb,
        url: item.url,
        previewVideo: item.previewVideo,
        duration: item.duration,
        watchedAt: Date.now()
      };
      const updated = [entry, ...prev];
      try {
        localStorage.setItem(
          'pyrmsearchRecentlyWatched',
          JSON.stringify(updated)
        );
      } catch {
        // ignore
      }
      return updated.slice(0, 50);
    });
  }, []);

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

  // Add video to a collection
  const handleAddToCollection = (collectionId: string, item: VideoResult) => {
    setCollections(prev => {
      const updated = prev.map(c => {
        if (c.id !== collectionId) return c;
        if (c.videoIds.includes(item.url)) return c;
        return { ...c, videoIds: [...c.videoIds, item.url] };
      });
      try {
        localStorage.setItem('pyrmsearchCollections', JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
    showToast('Added to collection');
  };

  // Remove video from a collection
  const handleRemoveFromCollection = (
    collectionId: string,
    videoUrl: string
  ) => {
    setCollections(prev => {
      const updated = prev.map(c => {
        if (c.id !== collectionId) return c;
        return { ...c, videoIds: c.videoIds.filter(id => id !== videoUrl) };
      });
      try {
        localStorage.setItem('pyrmsearchCollections', JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
    showToast('Removed from collection');
  };

  // Export all data to JSON
  const handleExportData = () => {
    const data = {
      favorites,
      screenshots,
      history,
      recentlyWatched,
      collections,
      exportedAt: Date.now()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pyrmsearch-backup-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Data exported successfully!');
    addLog('💾 Exported pyrmsearch data backup');
  };

  // Import data from JSON file
  const handleImportData = async (file: File) => {
    setIsImporting(true);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      if (data.favorites && Array.isArray(data.favorites)) {
        setFavorites(data.favorites);
        localStorage.setItem(
          'pyrmsearchFavorites',
          JSON.stringify(data.favorites)
        );
      }
      if (data.screenshots && Array.isArray(data.screenshots)) {
        setScreenshots(data.screenshots);
        localStorage.setItem(
          'pyrmsearchScreenshots',
          JSON.stringify(data.screenshots)
        );
      }
      if (data.history && Array.isArray(data.history)) {
        setHistory(data.history);
        localStorage.setItem('searchHistory', JSON.stringify(data.history));
      }
      if (data.recentlyWatched && Array.isArray(data.recentlyWatched)) {
        setRecentlyWatched(data.recentlyWatched);
        localStorage.setItem(
          'pyrmsearchRecentlyWatched',
          JSON.stringify(data.recentlyWatched)
        );
      }
      if (data.collections && Array.isArray(data.collections)) {
        setCollections(data.collections);
        localStorage.setItem(
          'pyrmsearchCollections',
          JSON.stringify(data.collections)
        );
      }
      setIsImporting(false);
      setImportFile(null);
      showToast('Data imported successfully!');
      addLog('📥 Imported pyrmsearch data backup');
    } catch (err) {
      setIsImporting(false);
      console.error('Import failed', err);
      showToast('Import failed — invalid file');
    }
  };

  // Share via termux-share or Web Share API
  const handleShareVideo = (item: VideoResult) => {
    const shareData = {
      title: item.title,
      text: `Check out "${item.title}" on ${item.siteName} via pyrmsearch`,
      url: item.url
    };
    if (navigator.share) {
      navigator.share(shareData).catch(() => {
        // ignore
      });
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(item.url);
      showToast('Link copied to clipboard');
    }
  };

  // Handle share action for VideoCard
  const handleShare = (item: VideoResult) => {
    const shareData = {
      title: item.title,
      text: `Check out "${item.title}" on ${item.siteName} via pyrmsearch`,
      url: item.url
    };

    // Web Share API (mobile browsers)
    if (navigator.share) {
      navigator.share(shareData).catch(() => {
        // Fallback to clipboard if share fails
        navigator.clipboard.writeText(item.url).then(() => {
          showToast('Link copied to clipboard');
        });
      });
      return;
    }

    // Fallback: copy to clipboard
    navigator.clipboard.writeText(item.url).then(() => {
      showToast('Link copied to clipboard');
    });
  };

  // Share screenshot data URL
  const handleShareScreenshot = (shot: SavedScreenshot) => {
    if (navigator.share) {
      navigator
        .share({
          title: shot.title,
          text: `Screenshot from ${shot.siteName} via pyrmsearch`,
          files: []
        })
        .catch(() => {
          // ignore
        });
    } else {
      showToast('Share not available on this device');
    }
  };

  // Handle direct WebAPK 1-tap installation
  const handleInstallApp = async () => {
    const success = await install();
    if (success) {
      showToast('pyrmsearch installed successfully!');
      addLog('📲 pyrmsearch installed to device home screen');
      setShowApkModal(false);
    }
  };

  // Execute Search / Hunt
  const handleExecute = async (
    targetPage: number = 1,
    overrideQuery?: string,
    overrideSite?: string
  ) => {
    if (loadingRef.current) return;

    const q = (overrideQuery !== undefined ? overrideQuery : query).trim();
    const currentSite = overrideSite || site;
    if (overrideSite && overrideSite !== site) {
      setSite(overrideSite);
      setPage(1);
      seenUrls.current.clear();
      setHasMore(true);
    }

    const cfg: SiteConfig = SITES[currentSite] || SITES.xvideos;

    if (!q) {
      showToast('Please enter a search query');
      return;
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
    } catch (error: any) {
      console.error(error);
      addLog(`Hunt failed: ${error.message || 'Unknown error'}`);
      showToast('Search encountered an error. Check console.');
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
    } catch (err: any) {
      addLog(`Site tests error: ${err.message}`);
    } finally {
      setTestingSites(false);
    }
  };

  // Test single site
  const handleTestSingleSite = async (siteId: string) => {
    const cfg = SITES[siteId];
    if (!cfg) return;
    addLog(`⚡ Pinging ${cfg.name}...`);
    const start = Date.now();
    try {
      const res = await fetch(`/api/scrape?site=${siteId}&query=test&page=1`);
      const latency = Date.now() - start;
      const data = await res.json();
      const isOk = res.ok && data.success;
      setSiteTests(prev => {
        const other = prev.filter(s => s.id !== siteId);
        return [
          ...other,
          {
            id: siteId,
            name: cfg.name,
            base: cfg.base,
            icon: cfg.icon,
            color: cfg.color,
            status: res.status,
            ok: isOk,
            latencyMs: latency
          }
        ];
      });
      addLog(`⚡ ${cfg.name}: ${res.status} (${latency}ms)`);
      showToast(`${cfg.name}: ${isOk ? 'Online' : 'Failed'} (${latency}ms)`);
    } catch (e: any) {
      const latency = Date.now() - start;
      setSiteTests(prev => {
        const other = prev.filter(s => s.id !== siteId);
        return [
          ...other,
          {
            id: siteId,
            name: cfg.name,
            base: cfg.base,
            icon: cfg.icon,
            color: cfg.color,
            status: 0,
            ok: false,
            latencyMs: latency,
            error: e.message
          }
        ];
      });
      addLog(`⚡ ${cfg.name} failed: ${e.message}`);
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

      {/* APK & App Installation Modal */}
      {showApkModal && (
        <div
          onClick={() => setShowApkModal(false)}
          className='fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-200'
        >
          <div
            onClick={e => e.stopPropagation()}
            className='bg-[#0a0a14] border border-[#05d9e8] rounded-2xl max-w-xl w-full overflow-hidden shadow-[0_0_35px_rgba(5,217,232,0.35)] relative'
          >
            {/* Modal Header */}
            <div className='p-4 bg-[#070710] border-b border-[#1a1a2e] flex items-center justify-between'>
              <div className='flex items-center gap-2.5'>
                <div className='w-8 h-8 rounded-lg bg-gradient-to-br from-[#ff2a6d] to-[#05d9e8] flex items-center justify-center text-white text-xs font-black'>
                  PY
                </div>
                <div>
                  <h3 className='text-sm font-extrabold text-white tracking-wider flex items-center gap-1.5'>
                    <span>GET PYRMSEARCH APK & APP</span>
                  </h3>
                  <p className='text-[10px] text-[#6a6a8a]'>
                    Android WebAPK • Offline Standalone • Zero Ads
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowApkModal(false)}
                className='text-[#6a6a8a] hover:text-white p-1 rounded-lg hover:bg-white/10 transition-colors cursor-pointer'
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Content */}
            <div className='p-5 space-y-4'>
              {/* Option 1: 1-Tap Direct WebAPK Installation */}
              <div className='bg-[#05050a] border border-[#ff2a6d]/40 rounded-xl p-4 relative overflow-hidden group'>
                <div className='flex items-start justify-between gap-3'>
                  <div>
                    <div className='flex items-center gap-2'>
                      <Zap size={16} className='text-[#ff2a6d]' />
                      <h4 className='text-sm font-bold text-white'>
                        Direct 1-Tap Install (WebAPK)
                      </h4>
                      <span className='px-1.5 py-0.5 rounded text-[9px] font-extrabold bg-[#ff2a6d]/20 text-[#ff2a6d] border border-[#ff2a6d]/40'>
                        RECOMMENDED
                      </span>
                    </div>
                    <p className='text-xs text-[#a0a0cf] mt-1.5 leading-relaxed'>
                      Installs <strong>pyrmsearch</strong> directly to your
                      Android device with fullscreen standalone launch, custom
                      app icon, and hardware acceleration without browser bars.
                    </p>
                  </div>
                </div>

                <div className='mt-3.5'>
                  {isInstalled ? (
                    <div className='flex items-center gap-2 text-xs font-bold text-emerald-400 py-1'>
                      <CheckCircle2 size={16} />
                      <span>Already installed as native standalone app!</span>
                    </div>
                  ) : isInstallable ? (
                    <button
                      onClick={handleInstallApp}
                      className='w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-[#ff2a6d] to-[#d300c5] hover:opacity-95 text-white font-extrabold text-xs tracking-wider shadow-[0_0_15px_rgba(255,42,109,0.5)] flex items-center justify-center gap-2 cursor-pointer transition-all'
                    >
                      <Smartphone size={16} />
                      <span>INSTALL PYRMSEARCH ON DEVICE</span>
                    </button>
                  ) : (
                    <div className='text-xs text-[#6a6a8a] bg-[#0c0c1a] p-2.5 rounded-lg border border-[#1a1a2e]'>
                      💡 Open on Android Chrome/Edge and tap the button, or use
                      Chrome menu (<strong>⋮</strong>) &rarr;{' '}
                      <strong>Install app</strong> or{' '}
                      <strong>Add to Home screen</strong>.
                    </div>
                  )}
                </div>
              </div>

              {/* Option 2: Direct pyrmsearch.apk File Download */}
              <div className='bg-[#05050a] border border-[#05d9e8]/40 rounded-xl p-4'>
                <div className='flex items-center gap-2'>
                  <ArrowDownToLine size={16} className='text-[#05d9e8]' />
                  <h4 className='text-sm font-bold text-white'>
                    Download Standalone pyrmsearch.apk
                  </h4>
                  <span className='px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#05d9e8]/20 text-[#05d9e8] border border-[#05d9e8]/40'>
                    APK FILE
                  </span>
                </div>
                <p className='text-xs text-[#a0a0cf] mt-1.5'>
                  Download the compiled Android package archive (
                  <code>pyrmsearch.apk</code>) to install or sideload directly.
                </p>

                <div className='mt-3 flex items-center justify-between gap-3 pt-2 border-t border-[#1a1a2e]'>
                  <div className='text-[11px] text-[#6a6a8a] font-mono'>
                    Package: com.pyrmsearch.app • v1.0.0
                  </div>
                  <a
                    href='/pyrmsearch.apk'
                    download='pyrmsearch.apk'
                    onClick={() => {
                      showToast('Downloading pyrmsearch.apk...');
                      addLog('📥 Download started: pyrmsearch.apk');
                    }}
                    className='py-2 px-5 rounded-lg bg-[#05d9e8] text-black font-extrabold text-xs tracking-wider hover:bg-[#00b4d8] shadow-[0_0_12px_rgba(5,217,232,0.4)] flex items-center gap-1.5 cursor-pointer transition-all'
                  >
                    <Download size={14} />
                    <span>DOWNLOAD APK</span>
                  </a>
                </div>
              </div>

              {/* iOS Note */}
              <div className='bg-[#070710] border border-[#1a1a2e] rounded-xl p-3 text-xs text-[#6a6a8a] flex items-center gap-2'>
                <ShieldCheck
                  size={16}
                  className='text-[#05d9e8] flex-shrink-0'
                />
                <span>
                  For iPhone / iPad users: Tap the <strong>Share</strong> icon
                  in Safari toolbar, then choose{' '}
                  <strong>Add to Home Screen</strong>.
                </span>
              </div>
            </div>
          </div>
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
                        setModalMuted(false);
                      }}
                      className='absolute bottom-4 left-4 bg-[#05d9e8] text-black px-3 py-1.5 rounded-full text-xs font-black tracking-wider shadow-[0_0_15px_rgba(5,217,232,0.6)] hover:opacity-90 transition-all cursor-pointer z-20 flex items-center gap-1.5'
                    >
                      <Play size={12} className='fill-current' />
                      <span>PLAY FULL VIDEO</span>
                    </button>
                  )}

                  {/* Back to preview badge */}
                  {modalUsingFull && (
                    <button
                      onClick={() => {
                        setModalVideoSrc(previewModalItem.previewVideo || '');
                        setModalUsingFull(false);
                        setModalVideoLoaded(false);
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
          onClick={() => setShowApkModal(true)}
          className='ml-auto hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#05d9e8]/10 border border-[#05d9e8]/40 text-[#05d9e8] text-xs font-bold hover:bg-[#05d9e8]/20 transition-all cursor-pointer'
        >
          <Download size={13} />
          <span>DOWNLOAD APK</span>
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
                  onChange={e => setSite(e.target.value)}
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
                  type='text'
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
                    className={`px-3 py-1.5 rounded-full text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                      selectedTrendingCategory?.id === cat.id
                        ? `bg-gradient-to-r from-${cat.color.replace('#', '')} text-white shadow-[0_0_10px_rgba(${parseInt(cat.color.slice(1, 3), 16)},${parseInt(cat.color.slice(3, 5), 16)},${parseInt(cat.color.slice(5, 7), 16)},0.4)]`
                        : 'bg-[#05050a] border border-[#1a1a2e] text-[#a0a0cf] hover:border-[#ff2a6d] hover:text-[#ff2a6d]'
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
                Ready for the hunt, Master.
              </p>
              <p className='text-xs text-[#6a6a8a] mt-2 max-w-md mx-auto'>
                Hover over video cards to watch live video teaser previews!
                Select from 13 sources like OK.XXX, Hello.Porn, Users.Porn,
                XVideos, and Eporner.
              </p>
              <button
                onClick={() => setShowApkModal(true)}
                className='mt-4 px-4 py-2 rounded-lg bg-[#05d9e8]/15 border border-[#05d9e8]/40 text-[#05d9e8] text-xs font-bold hover:bg-[#05d9e8]/25 transition-all inline-flex items-center gap-1.5 cursor-pointer'
              >
                <Smartphone size={14} />
                <span>Download pyrmsearch APK on Mobile</span>
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
                    onSaveScreenshot={item => {
                      // Find the video element inside this card to capture a frame
                      const cardEl = document.querySelector(
                        `[data-card-id="${item.id}"]`
                      );
                      const videoEl = cardEl?.querySelector('video');
                      const dataUrl = captureVideoFrame(videoEl || null);
                      if (dataUrl) handleSaveScreenshot(item, dataUrl);
                    }}
                    onShare={handleShare}
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
                  onClick={() => {
                    const links = favorites.map(f => f.url).join('\n');
                    navigator.clipboard.writeText(links);
                    showToast(`Copied ${favorites.length} URLs!`);
                    addLog(
                      `Copied all ${favorites.length} favorite links to clipboard`
                    );
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
                  onSaveScreenshot={item => {
                    const cardEl = document.querySelector(
                      `[data-card-id="${item.id}"]`
                    );
                    const videoEl = cardEl?.querySelector('video');
                    const dataUrl = captureVideoFrame(videoEl || null);
                    if (dataUrl) handleSaveScreenshot(item, dataUrl);
                  }}
                  onShare={handleShare}
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
                  onClick={() => {
                    const dataUrls = screenshots.map(s => s.dataUrl).join('\n');
                    navigator.clipboard.writeText(dataUrls);
                    showToast(
                      `Copied ${screenshots.length} screenshot data URLs!`
                    );
                    addLog(
                      `Copied all ${screenshots.length} screenshot data URLs`
                    );
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
                    <img
                      src={shot.dataUrl}
                      alt={shot.title}
                      className='w-full h-full object-cover transition-transform duration-300 group-hover:scale-105'
                    />
                    <div
                      style={{ backgroundColor: shot.siteColor }}
                      className='absolute bottom-2 left-2 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-black/85 text-white border z-10 backdrop-blur-xs'
                    >
                      {shot.siteName}
                    </div>
                    <button
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
                Videos you've opened in the teaser modal. Tap to re-open.
              </p>
            </div>
            {recentlyWatched.length > 0 && (
              <button
                onClick={() => {
                  setRecentlyWatched([]);
                  localStorage.removeItem('pyrmsearchRecentlyWatched');
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
                Open videos in the teaser modal to see them here.
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
                  className='bg-[#05050a] border border-[#1a1a2e] rounded-xl overflow-hidden flex flex-col cursor-pointer group hover:border-[#05d9e8]/50 transition-all'
                  onClick={() =>
                    setPreviewModalItem({
                      id: item.id,
                      title: item.title,
                      siteName: item.siteName,
                      siteColor: item.siteColor,
                      thumb: item.thumb,
                      url: item.url,
                      previewVideo: item.previewVideo,
                      duration: item.duration
                    } as VideoResult)
                  }
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
                onClick={handleClearFavorites}
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
                          favorites.find(f => f.url === url) ||
                          recentlyWatched.find(r => r.url === url);
                        if (!item) return null;
                        return (
                          <div
                            key={url}
                            className='bg-[#0a0a12] border border-[#1a1a2e] rounded-lg p-2 cursor-pointer hover:border-[#05d9e8]/50 transition-all group'
                            onClick={e => {
                              e.stopPropagation();
                              setPreviewModalItem(item as VideoResult);
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
                Errors ({Object.keys(SITES).length - onlineSiteCount})
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
