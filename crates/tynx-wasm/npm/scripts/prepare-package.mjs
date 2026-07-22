import { copyFile } from "node:fs/promises";

for (const license of ["LICENSE-APACHE", "LICENSE-MIT"]) {
  await copyFile(
    new URL(`../../../../${license}`, import.meta.url),
    new URL(`../${license}`, import.meta.url),
  );
}
