// PrismaClient singleton — Next.js dev mode hot-reloads modules, so without
// the global cache you'd accumulate connections every save. The pattern is
// from the Prisma docs and is the standard for any Next.js + Prisma app.
import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as { prisma?: PrismaClient };

export const prisma: PrismaClient = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
