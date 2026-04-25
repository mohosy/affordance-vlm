import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "affordance-vlm — temporal reasoning on construction body-cam footage",
  description:
    "Frontier VLMs see one frame at a time. Construction is temporal. We close the gap with a Qwen2.5-VL multi-frame LoRA fine-tune on Ironsite footage.",
  openGraph: {
    title: "affordance-vlm",
    description:
      "Frontier VLMs are fundamentally static. Construction is temporal. We close the gap.",
    url: "https://affordance-vlm.vercel.app",
    siteName: "affordance-vlm",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "affordance-vlm",
    description:
      "Frontier VLMs are fundamentally static. Construction is temporal. We close the gap.",
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
