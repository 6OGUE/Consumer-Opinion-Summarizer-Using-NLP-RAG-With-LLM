import { jsPDF } from 'jspdf';
import type { FinalInsightResponse } from '../types';

export async function exportInsightsToPDF(
  finalInsight: FinalInsightResponse,
  product: string,
  score: number | null
) {
  const pdf = new jsPDF('p', 'mm', 'a4');
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  let yPosition = 15;

  // Title
  pdf.setFontSize(24);
  pdf.text('Consumer Opinion Analysis', pageWidth / 2, yPosition, { align: 'center' });
  yPosition += 12;

  // Product name
  pdf.setFontSize(16);
  pdf.text(product.toUpperCase(), pageWidth / 2, yPosition, { align: 'center' });
  yPosition += 10;

  // Score
  if (score !== null) {
    pdf.setFontSize(12);
    pdf.text(`Overall Score: ${score}%`, pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 8;
  }

  // Divider
  pdf.setLineWidth(0.5);
  pdf.line(15, yPosition, pageWidth - 15, yPosition);
  yPosition += 8;

  // Helper function to add text with wrapping
  const addWrappedText = (text: string, fontSize: number, title?: string) => {
    if (yPosition > pageHeight - 20) {
      pdf.addPage();
      yPosition = 15;
    }

    if (title) {
      pdf.setFontSize(fontSize);
      pdf.setFont('helvetica', 'bold');
      pdf.text(title, 15, yPosition);
      yPosition += fontSize / 2 + 2;
      pdf.setFont('helvetica', 'normal');
    }

    pdf.setFontSize(fontSize);
    const lines = pdf.splitTextToSize(text, pageWidth - 30);
    pdf.text(lines, 15, yPosition);
    yPosition += lines.length * (fontSize / 2.5) + 5;
  };

  // Overview
  addWrappedText(finalInsight.overview, 10, 'Overview');

  // Highlights
  if (finalInsight.unique_features.length > 0) {
    if (yPosition > pageHeight - 20) {
      pdf.addPage();
      yPosition = 15;
    }
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Highlights', 15, yPosition);
    yPosition += 6;
    pdf.setFont('helvetica', 'normal');

    finalInsight.unique_features.forEach((feature) => {
      if (yPosition > pageHeight - 15) {
        pdf.addPage();
        yPosition = 15;
      }
      pdf.setFontSize(10);
      pdf.text(`• ${feature}`, 20, yPosition);
      yPosition += 5;
    });
    yPosition += 3;
  }

  // Strengths
  if (finalInsight.strengths.length > 0) {
    if (yPosition > pageHeight - 20) {
      pdf.addPage();
      yPosition = 15;
    }
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Strengths', 15, yPosition);
    yPosition += 6;
    pdf.setFont('helvetica', 'normal');

    finalInsight.strengths.forEach((strength) => {
      if (yPosition > pageHeight - 15) {
        pdf.addPage();
        yPosition = 15;
      }
      pdf.setFontSize(10);
      pdf.text(`• ${strength}`, 20, yPosition);
      yPosition += 5;
    });
    yPosition += 3;
  }

  // Weaknesses
  if (finalInsight.weaknesses.length > 0) {
    if (yPosition > pageHeight - 20) {
      pdf.addPage();
      yPosition = 15;
    }
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Weaknesses', 15, yPosition);
    yPosition += 6;
    pdf.setFont('helvetica', 'normal');

    finalInsight.weaknesses.forEach((weakness) => {
      if (yPosition > pageHeight - 15) {
        pdf.addPage();
        yPosition = 15;
      }
      pdf.setFontSize(10);
      pdf.text(`• ${weakness}`, 20, yPosition);
      yPosition += 5;
    });
    yPosition += 3;
  }

  // Alternatives
  if (finalInsight.alternatives.length > 0) {
    if (yPosition > pageHeight - 20) {
      pdf.addPage();
      yPosition = 15;
    }
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Alternatives', 15, yPosition);
    yPosition += 6;
    pdf.setFont('helvetica', 'normal');

    finalInsight.alternatives.forEach((alt) => {
      if (yPosition > pageHeight - 15) {
        pdf.addPage();
        yPosition = 15;
      }
      pdf.setFontSize(10);
      pdf.text(`• ${alt}`, 20, yPosition);
      yPosition += 5;
    });
    yPosition += 3;
  }

  // Final Insight
  addWrappedText(finalInsight.final_insight, 10, 'Final Insight');

  // Generate filename and download
  const filename = `${product.replace(/\s+/g, '_')}_insights.pdf`;
  pdf.save(filename);
}