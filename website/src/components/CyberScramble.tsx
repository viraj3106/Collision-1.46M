import React, { useState, useRef, useEffect } from 'react';

interface CyberScrambleProps {
  text: string;
  className?: string;
  scrambleOnHover?: boolean;
  triggerKey?: any;
}

const CHARS = '0123456789ABCDEF0123456789!@#$%^&*<>[]{}';

export const CyberScramble: React.FC<CyberScrambleProps> = ({
  text,
  className = '',
  scrambleOnHover = true,
  triggerKey,
}) => {
  const [displayText, setDisplayText] = useState(text);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startScramble = () => {
    let iteration = 0;
    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      setDisplayText(() =>
        text
          .split('')
          .map((char, index) => {
            if (char === ' ' || char === '·' || char === '·' || char === '.') return char;
            if (index < iteration) {
              return text[index];
            }
            return CHARS[Math.floor(Math.random() * CHARS.length)];
          })
          .join('')
      );

      if (iteration >= text.length) {
        if (intervalRef.current) clearInterval(intervalRef.current);
      }

      iteration += 1 / 2;
    }, 28);
  };

  useEffect(() => {
    if (triggerKey !== undefined) {
      startScramble();
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [triggerKey, text]);

  return (
    <span
      className={`cyber-scramble-text ${className}`}
      onMouseEnter={() => scrambleOnHover && startScramble()}
      style={{ display: 'inline-block', fontVariantNumeric: 'tabular-nums' }}
    >
      {displayText}
    </span>
  );
};
