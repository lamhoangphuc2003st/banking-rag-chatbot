import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vietcombank RAG",
  description: "Grounded chatbot for public Vietcombank information"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
