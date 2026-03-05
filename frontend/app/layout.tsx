import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Quote Agent Dashboard",
  description: "Multi-Agent InsurTech Pipeline",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
