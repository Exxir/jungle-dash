-- Adds comparison_notes column to Jun table for audit trail
ALTER TABLE public."Jun"
    ADD COLUMN IF NOT EXISTS "comparison_notes" TEXT;
