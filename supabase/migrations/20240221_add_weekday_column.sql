-- Adds weekday column derived from date
ALTER TABLE public."Jun"
    ADD COLUMN IF NOT EXISTS "weekday" TEXT;

UPDATE public."Jun"
SET "weekday" = TRIM(TO_CHAR("date", 'FMDay'));
