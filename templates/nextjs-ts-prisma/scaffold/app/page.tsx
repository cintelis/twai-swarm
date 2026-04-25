// Server Component example: queries the DB at request time and renders
// the result as plain HTML. No client JS needed for the initial render.
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

export default async function Home() {
  const items = await prisma.item.findMany({ orderBy: { createdAt: "desc" } });
  return (
    <main>
      <h1>Items</h1>
      <ul>
        {items.map((item) => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
      {items.length === 0 ? <p>No items yet. POST to /api/items to add one.</p> : null}
    </main>
  );
}
