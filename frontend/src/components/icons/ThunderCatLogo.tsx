// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { useTheme } from "@kui/react";
import tcatLogoBlack from "../../assets/tcat-logo-black.png";
import tcatLogoWhite from "../../assets/tcat-logo-white.png";

interface ThunderCatLogoProps {
  height?: string;
  alt?: string;
  className?: string;
}

export default function ThunderCatLogo({
  height = "14px",
  alt = "ThunderCat Logo",
  className = "",
}: ThunderCatLogoProps): React.ReactElement {
  const { theme } = useTheme();
  const src = theme === "light" ? tcatLogoBlack : tcatLogoWhite;

  return (
    <img
      className={`tcat-logo-element ${className}`}
      data-testid="tcat-logo-element"
      src={src}
      alt={alt}
      height={height}
      style={{ display: "block", width: "auto" }}
    />
  );
}
