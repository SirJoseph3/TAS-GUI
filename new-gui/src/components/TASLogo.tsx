import React from 'react';

interface TASLogoProps {
  className?: string;
  size?: number;
}

export const TASLogo: React.FC<TASLogoProps> = ({ className = '', size = 20 }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Arka plan daire */}
      <circle cx="50" cy="50" r="48" fill="#1a1a1a" stroke="#333" strokeWidth="2" />
      
      {/* Üst üçgen - Kırmızı */}
      <path
        d="M50 15 L75 55 L50 45 L25 55 Z"
        fill="#dc2626"
      />
      
      {/* Alt üçgen - Kırmızı */}
      <path
        d="M50 85 L75 45 L50 55 L25 45 Z"
        fill="#dc2626"
      />
      
      {/* Orta beyaz çizgi */}
      <rect x="48" y="35" width="4" height="30" fill="#ffffff" rx="1" />
      
      {/* Yan çizgiler - stil için */}
      <path
        d="M30 50 L45 42"
        stroke="#dc2626"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M70 50 L55 42"
        stroke="#dc2626"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
};

export default TASLogo;
