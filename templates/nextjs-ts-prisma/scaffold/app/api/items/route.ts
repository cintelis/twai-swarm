// Route Handler example: GET lists all items, POST creates one. The body
// validation is intentionally minimal — replace with Zod (or your preferred
// validator) in real apps. Pattern shown is the App Router convention.
import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

export async function GET() {
  const items = await prisma.item.findMany({ orderBy: { createdAt: "desc" } });
  return NextResponse.json(items);
}

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const name = (body as { name?: unknown }).name;
  if (typeof name !== "string" || name.trim() === "") {
    return NextResponse.json({ error: "field 'name' is required" }, { status: 422 });
  }
  const item = await prisma.item.create({ data: { name } });
  return NextResponse.json(item, { status: 201 });
}
