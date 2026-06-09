// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import tcatLogo from "../../assets/tcat-logo.png";

interface ThunderCatLogoProps {
  height?: string;
  alt?: string;
  className?: string;
}

export default function ThunderCatLogo({
  height = "14px",
  alt = "ThunderCat Technology",
  className = "",
}: ThunderCatLogoProps): React.ReactElement {
  return (
    <img
      className={`tcat-logo-element ${className}`}
      data-testid="tcat-logo-element"
      src={tcatLogo}
      alt={alt}
      height={height}
      style={{ display: "block", width: "auto" }}
    />
  );
}
