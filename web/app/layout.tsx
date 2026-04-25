import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "affordance-vlm — part-level affordance grounding for embodied AI",
  description:
    "Frontier VLMs fail at telling a robot which part of an object to grip. We close the gap with a Qwen2.5-VL LoRA fine-tune on HOVA-500K.",
  openGraph: {
    title: "affordance-vlm",
    description:
      "Frontier VLMs fail at part-level affordance grounding. We close the gap.",
    url: "https://affordance-vlm.vercel.app",
    siteName: "affordance-vlm",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "affordance-vlm",
    description:
      "Frontier VLMs fail at part-level affordance grounding. We close the gap.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
