import React, { useState, useEffect, useRef } from 'react';

interface CollisionWordmarkProps {
  text?: string;
  fontSize?: string | number;
  className?: string;
  // Mode: 'touch-only' (middle hero) or 'auto-loop' (header / footer)
  mode: 'touch-only' | 'auto-loop';
  intervalSeconds?: number;
}

/**
 * High-definition, crisp 3D letter animation for "COLLISION":
 * - mode="touch-only" (Middle Hero): 3D turn forward and backward ONLY when touched/hovered by user.
 * - mode="auto-loop" (Header & Footer): Automated periodic wave cycle on extended timer.
 * - Razor-sharp text rendering without blur or empty white spaces.
 */
export const CollisionWordmark: React.FC<CollisionWordmarkProps> = ({
  text = 'COLLISION',
  fontSize,
  className = '',
  mode,
  intervalSeconds = 8,
}) => {
  const [activeLoopIndex, setActiveLoopIndex] = useState<number | null>(null);
  const [activeFlippingIndices, setActiveFlippingIndices] = useState<number[]>([]);
  const timeoutsRef = useRef<{ [key: number]: ReturnType<typeof setTimeout> }>({});

  // Automated wave cycle for Header & Footer
  useEffect(() => {
    if (mode !== 'auto-loop') return;

    let isMounted = true;
    let waveTimeout: ReturnType<typeof setTimeout>;
    const charCount = text.length;

    const runWaveCycle = () => {
      for (let i = 0; i < charCount; i++) {
        setTimeout(() => {
          if (isMounted) setActiveLoopIndex(i);
        }, i * 140);
      }

      setTimeout(() => {
        if (isMounted) setActiveLoopIndex(null);
      }, charCount * 140 + 700);

      waveTimeout = setTimeout(runWaveCycle, intervalSeconds * 1000);
    };

    waveTimeout = setTimeout(runWaveCycle, 2000);

    return () => {
      isMounted = false;
      clearTimeout(waveTimeout);
    };
  }, [mode, text, intervalSeconds]);

  // Touch & Hover handler for Middle Hero
  const triggerLetterTurn = (index: number) => {
    if (mode !== 'touch-only') return;

    if (timeoutsRef.current[index]) {
      clearTimeout(timeoutsRef.current[index]);
    }

    setActiveFlippingIndices(prev => (prev.includes(index) ? prev : [...prev, index]));

    // Automatically complete turn forward and return backward smoothly
    timeoutsRef.current[index] = setTimeout(() => {
      setActiveFlippingIndices(prev => prev.filter(i => i !== index));
    }, 650);
  };

  const handleWordWave = () => {
    if (mode !== 'touch-only') return;
    text.split('').forEach((_, idx) => {
      setTimeout(() => {
        triggerLetterTurn(idx);
      }, idx * 60);
    });
  };

  return (
    <span
      className={`collision-wordmark ${className}`}
      onClick={handleWordWave}
      onTouchStart={handleWordWave}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        perspective: '1200px',
        fontWeight: 900,
        letterSpacing: '-0.035em',
        fontSize: fontSize || 'inherit',
        userSelect: 'none',
        lineHeight: 1,
        cursor: mode === 'touch-only' ? 'pointer' : 'default',
      }}
    >
      {text.split('').map((char, index) => {
        const isFlipping = activeFlippingIndices.includes(index);
        const isLooping = activeLoopIndex === index;

        return (
          <span
            key={index}
            className={`collision-letter ${isFlipping ? 'letter-touch-flip' : ''} ${isLooping ? 'letter-auto-wave' : ''}`}
            onMouseEnter={(e) => {
              e.stopPropagation();
              triggerLetterTurn(index);
            }}
            onTouchStart={(e) => {
              e.stopPropagation();
              triggerLetterTurn(index);
            }}
          >
            {char}
          </span>
        );
      })}
    </span>
  );
};
