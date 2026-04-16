"""
Class-specific prompts for anomaly detection.

Each class has exactly 14 normal prompts and 10 abnormal prompts,
matching the original prompt count (7 states × 2 templates = 14 normal,
5 states × 2 templates = 10 abnormal) so that encode_text_with_prompt_ensemble
reshape logic remains compatible.

Prompts are designed based on actual defect types in MVTec AD and ViSA datasets,
providing more discriminative text-image alignment than generic "damaged/flawless".
"""

NUM_NORMAL_PROMPTS = 14
NUM_ABNORMAL_PROMPTS = 10

# ============================================================
# MVTec AD Classes (15 classes)
# ============================================================

class_specific_prompts = {

    # ----------------------------------------------------------
    # BOTTLE — defects: broken_large, broken_small, contamination
    # ----------------------------------------------------------
    'bottle': {
        'normal': [
            'a photo of a defect-free bottle with uniform transparent surface.',
            'a photo of a bottle with clear glass and no visible scratches.',
            'a photo of a flawless bottle with intact shape and smooth edges.',
            'a photo of a perfect bottle with consistent transparency throughout.',
            'a photo of a bottle without any cracks or chips on the body.',
            'a photo of an unblemished bottle with clean interior and no particles.',
            'a photo of a bottle showing no contamination or foreign material inside.',
            'a photo of a normal bottle with properly formed neck and base.',
            'a photo of a bottle with even surface finish and no fracture lines.',
            'a photo of a bottle with uniform glass thickness and no cloudiness.',
            'a photo of a pristine bottle with no broken or chipped regions.',
            'a photo of a bottle in perfect condition with smooth contours.',
            'a photo of a good bottle with no missing fragments or holes.',
            'a photo of a bottle with clean surface free of any blemishes or stains.',
        ],
        'abnormal': [
            'a photo of a bottle with a visible crack running across the glass surface.',
            'a photo of a broken bottle with large missing fragments on the body.',
            'a photo of a bottle with small chips or broken pieces near the edge.',
            'a photo of a bottle with contamination or dark foreign particles inside.',
            'a photo of a damaged bottle with fracture lines radiating from an impact point.',
            'a photo of a bottle with a hole or puncture in the glass wall.',
            'a photo of a bottle with surface scratches and visible scuff marks.',
            'a photo of a bottle with discoloration or milky staining on the glass.',
            'a photo of a bottle with an irregular broken rim or jagged edge.',
            'a photo of a defective bottle with embedded debris or foreign material.',
        ],
    },

    # ----------------------------------------------------------
    # CABLE — defects: bent_wire, cable_swap, combined, cut_inner,
    #         cut_outer, missing_cable, missing_wire, poke_insulation
    # ----------------------------------------------------------
    'cable': {
        'normal': [
            'a photo of a defect-free cable with evenly arranged internal wires.',
            'a photo of a cable with intact outer insulation and no cuts.',
            'a photo of a flawless cable cross-section showing properly aligned conductors.',
            'a photo of a perfect cable with all wires present in correct positions.',
            'a photo of a cable without any bent or displaced wires.',
            'a photo of an unblemished cable with smooth outer sheath and no tears.',
            'a photo of a cable showing complete insulation coverage on all conductors.',
            'a photo of a normal cable with uniform color coding on each wire.',
            'a photo of a cable with consistent wire spacing and no gaps.',
            'a photo of a cable with all internal components properly seated.',
            'a photo of a pristine cable with no exposed copper or damaged shielding.',
            'a photo of a cable in perfect condition with intact protective coating.',
            'a photo of a good cable with no missing wires or swapped positions.',
            'a photo of a cable with clean cross-section showing proper wire arrangement.',
        ],
        'abnormal': [
            'a photo of a damaged cable.',
            'a photo of the damaged cable.',
            'a photo of a broken cable.',
            'a photo of the broken cable.',
            'a photo of a cable with a bent or kinked internal wire.',
            'a photo of a cable with wires swapped into incorrect positions.',
            'a photo of a cable with cut or sliced inner insulation exposing conductors.',
            'a photo of a cable with one or more missing wires in the cross-section.',
            'a photo of a cable with exposed copper due to stripped insulation.',
            'a photo of a cable with crushed or deformed internal structure.',
        ],
    },

    # ----------------------------------------------------------
    # CAPSULE — defects: crack, faulty_imprint, poke, scratch, squeeze
    # ----------------------------------------------------------
    'capsule': {
        'normal': [
            'a photo of a defect-free capsule with smooth uniform surface.',
            'a photo of a capsule with clear and correctly printed label markings.',
            'a photo of a flawless capsule with consistent color and no discoloration.',
            'a photo of a perfect capsule with intact seam and proper shape.',
            'a photo of a capsule without any cracks or splits on the shell.',
            'a photo of an unblemished capsule with no scratches or surface marks.',
            'a photo of a capsule showing even surface texture with no indentations.',
            'a photo of a normal capsule with properly aligned two halves.',
            'a photo of a capsule with smooth gelatin coating and no poke holes.',
            'a photo of a capsule with uniform cylindrical shape and no deformation.',
            'a photo of a pristine capsule with legible and complete imprint.',
            'a photo of a capsule in perfect condition for pharmaceutical inspection.',
            'a photo of a good capsule with no squeeze marks or compressed areas.',
            'a photo of a capsule with intact shell free of any visible defects.',
        ],
        'abnormal': [
            'a photo of a damaged capsule.',
            'a photo of the damaged capsule.',
            'a photo of a broken capsule.',
            'a photo of the broken capsule.',
            'a photo of a capsule with a visible crack along the body.',
            'a photo of a capsule with a faulty or misaligned printed imprint.',
            'a photo of a capsule with a poke hole punctured through the shell.',
            'a photo of a capsule with scratch lines across the surface.',
            'a photo of a capsule with squeeze deformation changing its shape.',
            'a photo of a capsule with dented or crushed surface areas.',
        ],
    },

    # ----------------------------------------------------------
    # CARPET — defects: color, cut, hole, metal_contamination, thread
    # ----------------------------------------------------------
    'carpet': {
        'normal': [
            'a photo of a defect-free carpet with uniform woven texture.',
            'a photo of a carpet with consistent color and no stains.',
            'a photo of a flawless carpet with evenly spaced threads throughout.',
            'a photo of a perfect carpet with no cuts or tears in the fabric.',
            'a photo of a carpet without any holes or missing fiber sections.',
            'a photo of an unblemished carpet with clean surface and no contamination.',
            'a photo of a carpet showing regular weave pattern with no loose threads.',
            'a photo of a normal carpet with uniform pile height across the surface.',
            'a photo of a carpet with consistent texture and no foreign objects embedded.',
            'a photo of a carpet with smooth surface free of wrinkles or folds.',
            'a photo of a pristine carpet with no metal particles or debris.',
            'a photo of a carpet in perfect condition with intact fiber structure.',
            'a photo of a good carpet with no color variations or bleached areas.',
            'a photo of a carpet with regular pattern and no visible defects.',
        ],
        'abnormal': [
            'a photo of a carpet with abnormal color patches or discoloration.',
            'a photo of a carpet with a visible cut or slash through the fabric.',
            'a photo of a carpet with a hole where fibers are completely missing.',
            'a photo of a carpet with metal contamination particles embedded in the surface.',
            'a photo of a carpet with loose or protruding threads disrupting the pattern.',
            'a photo of a carpet with a torn section showing frayed edges.',
            'a photo of a carpet with stained or bleached irregular regions.',
            'a photo of a carpet with foreign material stuck to the surface.',
            'a photo of a carpet with uneven texture from pulled or broken fibers.',
            'a photo of a defective carpet with visible weaving irregularities.',
        ],
    },

    # ----------------------------------------------------------
    # GRID — defects: bent, broken, glue, metal_contamination, thread
    # ----------------------------------------------------------
    'grid': {
        'normal': [
            'a photo of a defect-free grid with straight and evenly spaced lines.',
            'a photo of a grid with uniform mesh pattern and consistent gaps.',
            'a photo of a flawless grid with no bent or broken wires.',
            'a photo of a perfect grid with clean surface free of adhesive residue.',
            'a photo of a grid without any metal contamination or foreign particles.',
            'a photo of an unblemished grid with properly aligned horizontal and vertical bars.',
            'a photo of a grid showing consistent wire thickness throughout.',
            'a photo of a normal grid with no loose threads or fiber contamination.',
            'a photo of a grid with regular square openings and no deformation.',
            'a photo of a grid with clean metallic surface and no glue marks.',
            'a photo of a pristine grid with symmetrical pattern and no distortion.',
            'a photo of a grid in perfect condition for industrial quality inspection.',
            'a photo of a good grid with intact structure and no missing sections.',
            'a photo of a grid with uniform appearance and no visible anomalies.',
        ],
        'abnormal': [
            'a photo of a grid with bent or curved wire sections.',
            'a photo of a grid with broken wires leaving gaps in the mesh.',
            'a photo of a grid with glue or adhesive residue on the surface.',
            'a photo of a grid with metal contamination particles stuck to the wires.',
            'a photo of a grid with thread or fiber contamination caught in the mesh.',
            'a photo of a grid with misaligned bars creating irregular openings.',
            'a photo of a grid with crushed or deformed mesh sections.',
            'a photo of a grid with missing wire segments in the pattern.',
            'a photo of a grid with dark spots from foreign material contamination.',
            'a photo of a defective grid with visible structural damage to the mesh.',
        ],
    },

    # ----------------------------------------------------------
    # HAZELNUT — defects: crack, cut, hole, print
    # ----------------------------------------------------------
    'hazelnut': {
        'normal': [
            'a photo of a defect-free hazelnut with smooth brown shell surface.',
            'a photo of a hazelnut with natural uniform color and no cracks.',
            'a photo of a flawless hazelnut with intact rounded shape.',
            'a photo of a perfect hazelnut with consistent shell texture.',
            'a photo of a hazelnut without any cuts or puncture marks.',
            'a photo of an unblemished hazelnut with no holes or missing shell pieces.',
            'a photo of a hazelnut showing natural surface pattern with no damage.',
            'a photo of a normal hazelnut with clean surface and no foreign marks.',
            'a photo of a hazelnut with complete shell coverage and no exposed kernel.',
            'a photo of a hazelnut with typical brown coloring and no dark spots.',
            'a photo of a pristine hazelnut with no print marks or stains on the shell.',
            'a photo of a hazelnut in perfect condition with smooth outer surface.',
            'a photo of a good hazelnut with no visible cracks or fracture lines.',
            'a photo of a hazelnut with natural appearance free of any defects.',
        ],
        'abnormal': [
            'a photo of a hazelnut with a visible crack splitting the shell.',
            'a photo of a hazelnut with a cut or gouge mark on the surface.',
            'a photo of a hazelnut with a hole through the shell exposing the interior.',
            'a photo of a hazelnut with abnormal print marks or ink stains on the shell.',
            'a photo of a hazelnut with multiple fracture lines across the surface.',
            'a photo of a hazelnut with a deep scratch or carved line.',
            'a photo of a hazelnut with a broken or missing shell section.',
            'a photo of a hazelnut with dark discolored patches indicating damage.',
            'a photo of a hazelnut with surface erosion or pitted areas.',
            'a photo of a defective hazelnut with irregular shape from physical damage.',
        ],
    },

    # ----------------------------------------------------------
    # LEATHER — defects: color, cut, fold, glue, poke
    # ----------------------------------------------------------
    'leather': {
        'normal': [
            'a photo of a defect-free leather surface with uniform grain texture.',
            'a photo of leather with consistent dark color and no stains.',
            'a photo of a flawless leather piece with smooth even surface.',
            'a photo of a perfect leather surface with natural grain pattern.',
            'a photo of leather without any cuts or slash marks.',
            'a photo of an unblemished leather surface with no fold creases.',
            'a photo of leather showing uniform color distribution with no patches.',
            'a photo of a normal leather piece with no glue residue or marks.',
            'a photo of leather with intact surface and no poke holes.',
            'a photo of leather with consistent texture and no wrinkles or tears.',
            'a photo of a pristine leather surface with no visible damage.',
            'a photo of leather in perfect condition for material quality inspection.',
            'a photo of good leather with no discoloration or bleached areas.',
            'a photo of leather with clean flat surface free of any defects.',
        ],
        'abnormal': [
            'a photo of leather with abnormal color patches or discoloration.',
            'a photo of leather with a visible cut or slash through the surface.',
            'a photo of leather with fold marks or deep creases deforming the surface.',
            'a photo of leather with glue or adhesive residue staining the surface.',
            'a photo of leather with a poke hole punctured through the material.',
            'a photo of leather with torn or frayed edges from physical damage.',
            'a photo of leather with scratches or scuff marks on the grain surface.',
            'a photo of leather with wrinkled or buckled areas disrupting smoothness.',
            'a photo of leather with dark spots or stains from contamination.',
            'a photo of defective leather with multiple surface imperfections visible.',
        ],
    },

    # ----------------------------------------------------------
    # METAL NUT — defects: bent, color, flip, scratch
    # ----------------------------------------------------------
    'metal nut': {
        'normal': [
            'a photo of a defect-free metal nut with uniform hexagonal shape.',
            'a photo of a metal nut with consistent silver metallic color.',
            'a photo of a flawless metal nut with smooth machined surfaces.',
            'a photo of a perfect metal nut with properly formed threads.',
            'a photo of a metal nut without any scratches on the surface.',
            'a photo of an unblemished metal nut with correct orientation and no flipping.',
            'a photo of a metal nut showing even surface finish with no bending.',
            'a photo of a normal metal nut with symmetrical hexagonal edges.',
            'a photo of a metal nut with clean surface and no discoloration.',
            'a photo of a metal nut with uniform thread pattern and no burrs.',
            'a photo of a pristine metal nut with intact flat faces.',
            'a photo of a metal nut in perfect condition with sharp defined edges.',
            'a photo of a good metal nut with no bent or deformed sections.',
            'a photo of a metal nut with proper orientation and clean appearance.',
        ],
        'abnormal': [
            'a photo of a metal nut with bent or warped edges.',
            'a photo of a metal nut with abnormal color patches or rust stains.',
            'a photo of a metal nut placed upside down or flipped incorrectly.',
            'a photo of a metal nut with visible scratches across the surface.',
            'a photo of a metal nut with deformed hexagonal shape from bending.',
            'a photo of a metal nut with dark oxidation or discoloration marks.',
            'a photo of a metal nut with damaged threads or stripped sections.',
            'a photo of a metal nut with surface gouges or deep marks.',
            'a photo of a metal nut with incorrect orientation showing the wrong face.',
            'a photo of a defective metal nut with multiple surface defects visible.',
        ],
    },

    # ----------------------------------------------------------
    # PILL — defects: color, combined, contamination, crack,
    #        faulty_imprint, pill_type, scratch
    # ----------------------------------------------------------
    'pill': {
        'normal': [
            'a photo of a defect-free pill with smooth uniform coating.',
            'a photo of a pill with consistent white color and no spots.',
            'a photo of a flawless pill with clear and legible imprint text.',
            'a photo of a perfect pill with even rounded shape and smooth edges.',
            'a photo of a pill without any cracks or chips on the surface.',
            'a photo of an unblemished pill with no scratch marks or lines.',
            'a photo of a pill showing uniform color distribution with no contamination.',
            'a photo of a normal pill with proper imprint alignment and clarity.',
            'a photo of a pill with intact coating and no peeling or flaking.',
            'a photo of a pill with consistent tablet shape and correct dimensions.',
            'a photo of a pristine pill with no foreign particles on the surface.',
            'a photo of a pill in perfect condition for pharmaceutical quality control.',
            'a photo of a good pill with correct type identification markings.',
            'a photo of a pill with clean surface and properly formed edges.',
        ],
        'abnormal': [
            'a photo of a pill with abnormal color variation or discolored patches.',
            'a photo of a pill with combined defects including cracks and scratches.',
            'a photo of a pill with contamination spots or foreign particles on the surface.',
            'a photo of a pill with a visible crack or split across the tablet.',
            'a photo of a pill with faulty or smeared imprint markings.',
            'a photo of a pill of incorrect type mixed in with the wrong batch.',
            'a photo of a pill with scratch lines across the coating surface.',
            'a photo of a pill with chipped edges or broken tablet fragments.',
            'a photo of a pill with uneven coating showing bare spots.',
            'a photo of a defective pill with deformed shape or irregular surface.',
        ],
    },

    # ----------------------------------------------------------
    # SCREW — defects: manipulated_front, scratch_head, scratch_neck,
    #         thread_side, thread_top
    # ----------------------------------------------------------
    'screw': {
        'normal': [
            'a photo of a defect-free screw with clean head and sharp slot.',
            'a photo of a screw with uniform metallic surface and no scratches.',
            'a photo of a flawless screw with properly formed thread pattern.',
            'a photo of a perfect screw with consistent thread spacing throughout.',
            'a photo of a screw without any scratches on the head or neck.',
            'a photo of an unblemished screw with smooth shaft between head and threads.',
            'a photo of a screw showing uniform thread depth with no damage.',
            'a photo of a normal screw with properly shaped head and intact slot.',
            'a photo of a screw with clean neck area and no surface marks.',
            'a photo of a screw with consistent thread profile from top to tip.',
            'a photo of a pristine screw with no manipulated or altered features.',
            'a photo of a screw in perfect condition for hardware quality inspection.',
            'a photo of a good screw with sharp thread edges and clean grooves.',
            'a photo of a screw with intact surface finish and no visible defects.',
        ],
        'abnormal': [
            'a photo of a screw with a manipulated or tampered head surface.',
            'a photo of a screw with scratches on the head area.',
            'a photo of a screw with scratch marks on the neck between head and threads.',
            'a photo of a screw with damaged threads on the side of the shaft.',
            'a photo of a screw with damaged or deformed threads at the top.',
            'a photo of a screw with a stripped or worn head slot.',
            'a photo of a screw with irregular thread spacing indicating defects.',
            'a photo of a screw with surface gouges or dents on the shaft.',
            'a photo of a screw with bent or misaligned thread sections.',
            'a photo of a defective screw with visible manufacturing flaws.',
        ],
    },

    # ----------------------------------------------------------
    # TILE — defects: crack, glue_strip, gray_stroke, oil, rough
    # ----------------------------------------------------------
    'tile': {
        'normal': [
            'a photo of a defect-free tile with smooth uniform surface.',
            'a photo of a tile with consistent color and clean finish.',
            'a photo of a flawless tile with even glaze coating throughout.',
            'a photo of a perfect tile with no cracks or fracture lines.',
            'a photo of a tile without any oil stains or wet marks.',
            'a photo of an unblemished tile with no adhesive or glue residue.',
            'a photo of a tile showing uniform surface texture with no rough patches.',
            'a photo of a normal tile with clean edges and no gray strokes.',
            'a photo of a tile with consistent surface sheen and no dull spots.',
            'a photo of a tile with smooth glazed surface free of any marks.',
            'a photo of a pristine tile with no visible contamination or stains.',
            'a photo of a tile in perfect condition for construction quality check.',
            'a photo of a good tile with uniform appearance and no strip marks.',
            'a photo of a tile with clean polished surface and no defects.',
        ],
        'abnormal': [
            'a photo of a tile with a visible crack running across the surface.',
            'a photo of a tile with a glue strip or adhesive residue mark.',
            'a photo of a tile with a gray stroke or smear mark on the surface.',
            'a photo of a tile with oil stains or grease marks contaminating the surface.',
            'a photo of a tile with rough textured patches disrupting the smooth finish.',
            'a photo of a tile with chipped edges or corner damage.',
            'a photo of a tile with discolored streaks or lines on the glaze.',
            'a photo of a tile with surface bubbles or pitting in the coating.',
            'a photo of a tile with multiple hairline cracks forming a pattern.',
            'a photo of a defective tile with uneven surface and visible flaws.',
        ],
    },

    # ----------------------------------------------------------
    # TOOTHBRUSH — defects: defective bristles
    # ----------------------------------------------------------
    'toothbrush': {
        'normal': [
            'a photo of a defect-free toothbrush with evenly arranged bristles.',
            'a photo of a toothbrush with uniform bristle height and alignment.',
            'a photo of a flawless toothbrush with properly formed handle and head.',
            'a photo of a perfect toothbrush with neatly organized bristle tufts.',
            'a photo of a toothbrush without any missing or bent bristles.',
            'a photo of an unblemished toothbrush with clean and straight bristle rows.',
            'a photo of a toothbrush showing consistent bristle density across the head.',
            'a photo of a normal toothbrush with intact bristle pattern.',
            'a photo of a toothbrush with properly trimmed bristles at uniform length.',
            'a photo of a toothbrush with complete bristle coverage on the head.',
            'a photo of a pristine toothbrush with no manufacturing defects visible.',
            'a photo of a toothbrush in perfect condition for product quality control.',
            'a photo of a good toothbrush with symmetrical bristle arrangement.',
            'a photo of a toothbrush with clean appearance and no deformed bristles.',
        ],
        'abnormal': [
            'a photo of a toothbrush with missing bristle tufts leaving gaps.',
            'a photo of a toothbrush with bent or splayed bristles.',
            'a photo of a toothbrush with uneven bristle heights across the head.',
            'a photo of a toothbrush with discolored or contaminated bristles.',
            'a photo of a toothbrush with misaligned bristle rows.',
            'a photo of a toothbrush with clumped or fused bristles.',
            'a photo of a toothbrush with broken bristles scattered on the head.',
            'a photo of a toothbrush with irregular bristle pattern from defective molding.',
            'a photo of a toothbrush with extra or misplaced bristle tufts.',
            'a photo of a defective toothbrush with visibly damaged bristle arrangement.',
        ],
    },

    # ----------------------------------------------------------
    # TRANSISTOR — defects: bent_lead, cut_lead, damaged_case, misplaced
    # ----------------------------------------------------------
    'transistor': {
        'normal': [
            'a photo of a defect-free transistor with straight aligned leads.',
            'a photo of a transistor with intact plastic case and no damage.',
            'a photo of a flawless transistor with properly spaced and parallel leads.',
            'a photo of a perfect transistor with correct component placement.',
            'a photo of a transistor without any bent or misaligned leads.',
            'a photo of an unblemished transistor with clean case surface and clear markings.',
            'a photo of a transistor showing all three leads intact and straight.',
            'a photo of a normal transistor with properly formed lead tips.',
            'a photo of a transistor with undamaged case and no cracks or chips.',
            'a photo of a transistor with correct orientation on the circuit board.',
            'a photo of a pristine transistor with no cut or shortened leads.',
            'a photo of a transistor in perfect condition for electronics quality inspection.',
            'a photo of a good transistor with symmetrical lead arrangement.',
            'a photo of a transistor with clean leads and intact component body.',
        ],
        'abnormal': [
            'a photo of a damaged transistor.',
            'a photo of the damaged transistor.',
            'a photo of a broken transistor.',
            'a photo of the broken transistor.',
            'a photo of a transistor with a bent or damaged lead.',
            'a photo of a transistor with a cracked or damaged plastic case.',
            'a photo of a transistor with missing or cut lead sections.',
            'a photo of a transistor with melted or heat-damaged case material.',
            'a photo of a transistor with leads touching or bridging incorrectly.',
            'a photo of a transistor with visible physical defects or corrosion.',
        ],
    },

    # ----------------------------------------------------------
    # WOOD — defects: color, combined, hole, liquid, scratch
    # ----------------------------------------------------------
    'wood': {
        'normal': [
            'a photo of a defect-free wood surface with natural grain pattern.',
            'a photo of wood with consistent warm brown color and smooth finish.',
            'a photo of a flawless wood piece with uniform texture throughout.',
            'a photo of a perfect wood surface with no holes or punctures.',
            'a photo of wood without any scratches or gouge marks.',
            'a photo of an unblemished wood surface with clean grain lines.',
            'a photo of wood showing natural pattern with no stains or liquid marks.',
            'a photo of a normal wood piece with even color distribution.',
            'a photo of wood with smooth sanded surface and no rough areas.',
            'a photo of wood with intact finish and no peeling or chipping.',
            'a photo of a pristine wood surface with no discoloration or dark spots.',
            'a photo of wood in perfect condition for furniture quality inspection.',
            'a photo of good wood with natural grain and no visible defects.',
            'a photo of wood with clean surface free of any damage or contamination.',
        ],
        'abnormal': [
            'a photo of wood with abnormal color patches or dark discoloration.',
            'a photo of wood with combined defects including scratches and color changes.',
            'a photo of wood with a hole or knot hole penetrating the surface.',
            'a photo of wood with liquid stains or water damage marks.',
            'a photo of wood with visible scratch lines across the grain.',
            'a photo of wood with deep gouge marks or carved damage.',
            'a photo of wood with bleached or faded irregular areas.',
            'a photo of wood with ring stains from liquid contamination.',
            'a photo of wood with rough splintered sections on the surface.',
            'a photo of defective wood with multiple surface imperfections visible.',
        ],
    },

    # ----------------------------------------------------------
    # ZIPPER — defects: broken_teeth, combined, fabric_border,
    #          fabric_interior, rough, split_teeth, squeezed_teeth
    # ----------------------------------------------------------
    'zipper': {
        'normal': [
            'a photo of a defect-free zipper with evenly aligned metal teeth.',
            'a photo of a zipper with uniform teeth spacing and smooth fabric tape.',
            'a photo of a flawless zipper with properly interlocking teeth on both sides.',
            'a photo of a perfect zipper with intact fabric border and no fraying.',
            'a photo of a zipper without any broken or missing teeth.',
            'a photo of an unblemished zipper with clean stitching along the tape.',
            'a photo of a zipper showing consistent teeth height and alignment.',
            'a photo of a normal zipper with smooth gliding track and no obstructions.',
            'a photo of a zipper with intact fabric interior with no damage.',
            'a photo of a zipper with properly formed teeth and no rough edges.',
            'a photo of a pristine zipper with no split or squeezed teeth.',
            'a photo of a zipper in perfect condition for garment quality inspection.',
            'a photo of a good zipper with symmetrical teeth arrangement.',
            'a photo of a zipper with clean appearance and no visible defects.',
        ],
        'abnormal': [
            'a photo of a zipper with broken or missing teeth in the track.',
            'a photo of a zipper with combined defects on teeth and fabric.',
            'a photo of a zipper with damaged or frayed fabric along the border.',
            'a photo of a zipper with torn fabric in the interior tape area.',
            'a photo of a zipper with rough or jagged teeth edges.',
            'a photo of a zipper with split teeth that fail to interlock.',
            'a photo of a zipper with squeezed or compressed teeth out of alignment.',
            'a photo of a zipper with gaps where teeth are missing from the track.',
            'a photo of a zipper with bent teeth causing the track to jam.',
            'a photo of a defective zipper with visible manufacturing flaws in the teeth.',
        ],
    },

    # ============================================================
    # ViSA Dataset Classes (8 classes)
    # ============================================================

    # ----------------------------------------------------------
    # CANDLE — defects: surface anomalies, color spots, cracks
    # ----------------------------------------------------------
    'candle': {
        'normal': [
            'a photo of a defect-free candle with smooth uniform wax surface.',
            'a photo of a candle with consistent color and no spots or marks.',
            'a photo of a flawless candle with even cylindrical shape.',
            'a photo of a perfect candle with intact wick and clean surface.',
            'a photo of a candle without any cracks or fractures in the wax.',
            'a photo of an unblemished candle with smooth edges and no chips.',
            'a photo of a candle showing uniform wax texture with no bubbles.',
            'a photo of a normal candle with consistent opacity throughout.',
            'a photo of a candle with clean surface free of debris or contamination.',
            'a photo of a candle with proper shape and no deformation.',
            'a photo of a pristine candle with no discoloration or stains.',
            'a photo of a candle in perfect condition for product quality inspection.',
            'a photo of a good candle with smooth wax finish and no pitting.',
            'a photo of a candle with uniform appearance and intact structure.',
        ],
        'abnormal': [
            'a photo of a candle with visible cracks in the wax surface.',
            'a photo of a candle with discolored spots or foreign material embedded.',
            'a photo of a candle with surface pitting or air bubble holes.',
            'a photo of a candle with uneven wax distribution or lumps.',
            'a photo of a candle with chipped or broken edges.',
            'a photo of a candle with dark spots or stains on the surface.',
            'a photo of a candle with irregular shape from melting or deformation.',
            'a photo of a candle with scratches or gouge marks in the wax.',
            'a photo of a candle with contamination particles on the surface.',
            'a photo of a defective candle with visible manufacturing flaws.',
        ],
    },

    # ----------------------------------------------------------
    # CASHEW — defects: surface anomalies, cracks, spots
    # ----------------------------------------------------------
    'cashew': {
        'normal': [
            'a photo of a defect-free cashew with smooth natural curved surface.',
            'a photo of a cashew with uniform light brown color and no dark spots.',
            'a photo of a flawless cashew with intact kidney shape.',
            'a photo of a perfect cashew with consistent surface texture.',
            'a photo of a cashew without any cracks or splits in the shell.',
            'a photo of an unblemished cashew with no scratches or marks.',
            'a photo of a cashew showing natural surface with no discoloration.',
            'a photo of a normal cashew with even color tone throughout.',
            'a photo of a cashew with smooth surface and no rough patches.',
            'a photo of a cashew with intact edges and no broken pieces.',
            'a photo of a pristine cashew with no foreign material on the surface.',
            'a photo of a cashew in perfect condition for food quality inspection.',
            'a photo of a good cashew with natural appearance and no blemishes.',
            'a photo of a cashew with clean surface free of any visible defects.',
        ],
        'abnormal': [
            'a photo of a cashew with a visible crack or split in the surface.',
            'a photo of a cashew with dark discolored spots or burn marks.',
            'a photo of a cashew with irregular surface texture from damage.',
            'a photo of a cashew with broken or chipped sections.',
            'a photo of a cashew with mold or contamination on the surface.',
            'a photo of a cashew with scratch marks or surface abrasions.',
            'a photo of a cashew with unusual color patches indicating spoilage.',
            'a photo of a cashew with hole or puncture from insect damage.',
            'a photo of a cashew with rough or pitted surface areas.',
            'a photo of a defective cashew with visible quality defects.',
        ],
    },

    # ----------------------------------------------------------
    # CHEWINGGUM — defects: surface anomalies
    # ----------------------------------------------------------
    'chewinggum': {
        'normal': [
            'a photo of a defect-free chewing gum with smooth uniform surface.',
            'a photo of a chewing gum with consistent color and no marks.',
            'a photo of a flawless chewing gum with proper rectangular shape.',
            'a photo of a perfect chewing gum with even coating throughout.',
            'a photo of a chewing gum without any cracks or chips.',
            'a photo of an unblemished chewing gum with clean edges.',
            'a photo of a chewing gum showing uniform texture with no bubbles.',
            'a photo of a normal chewing gum with intact surface coating.',
            'a photo of a chewing gum with consistent thickness and shape.',
            'a photo of a chewing gum with smooth finish and no rough patches.',
            'a photo of a pristine chewing gum with no surface contamination.',
            'a photo of a chewing gum in perfect condition for quality inspection.',
            'a photo of a good chewing gum with no deformation or dents.',
            'a photo of a chewing gum with clean appearance and no visible defects.',
        ],
        'abnormal': [
            'a photo of a chewing gum with visible cracks on the surface.',
            'a photo of a chewing gum with discolored patches or spots.',
            'a photo of a chewing gum with chipped or broken corners.',
            'a photo of a chewing gum with surface bubbles or voids.',
            'a photo of a chewing gum with dents or compression marks.',
            'a photo of a chewing gum with foreign material on the surface.',
            'a photo of a chewing gum with uneven coating or bare spots.',
            'a photo of a chewing gum with scratches across the surface.',
            'a photo of a chewing gum with irregular shape from deformation.',
            'a photo of a defective chewing gum with visible manufacturing flaws.',
        ],
    },

    # ----------------------------------------------------------
    # FRYUM — defects: surface anomalies
    # ----------------------------------------------------------
    'fryum': {
        'normal': [
            'a photo of a defect-free fryum with uniform golden crispy surface.',
            'a photo of a fryum with consistent color and proper puffed shape.',
            'a photo of a flawless fryum with even texture throughout.',
            'a photo of a perfect fryum with intact edges and no broken pieces.',
            'a photo of a fryum without any dark burn spots or discoloration.',
            'a photo of an unblemished fryum with natural crispy appearance.',
            'a photo of a fryum showing uniform expansion with no collapsed areas.',
            'a photo of a normal fryum with consistent golden brown color.',
            'a photo of a fryum with proper shape and no deformation.',
            'a photo of a fryum with clean surface and no contamination.',
            'a photo of a pristine fryum with even frying and no raw spots.',
            'a photo of a fryum in perfect condition for food quality inspection.',
            'a photo of a good fryum with crispy texture and no soft areas.',
            'a photo of a fryum with natural appearance and no visible defects.',
        ],
        'abnormal': [
            'a photo of a fryum with dark burn spots or charred areas.',
            'a photo of a fryum with broken or crumbled sections.',
            'a photo of a fryum with abnormal color indicating under or over frying.',
            'a photo of a fryum with collapsed or flat areas that did not puff.',
            'a photo of a fryum with foreign material or contamination on the surface.',
            'a photo of a fryum with cracks or splits in the crispy surface.',
            'a photo of a fryum with irregular shape from production defects.',
            'a photo of a fryum with discolored patches or uneven frying.',
            'a photo of a fryum with holes or missing sections in the body.',
            'a photo of a defective fryum with visible quality issues.',
        ],
    },

    # ----------------------------------------------------------
    # MACARONI — defects: surface anomalies, cracks, spots
    # ----------------------------------------------------------
    'macaroni': {
        'normal': [
            'a photo of a defect-free macaroni with smooth tubular surface.',
            'a photo of a macaroni with uniform pale yellow color throughout.',
            'a photo of a flawless macaroni with proper curved elbow shape.',
            'a photo of a perfect macaroni with consistent wall thickness.',
            'a photo of a macaroni without any cracks or breaks in the pasta.',
            'a photo of an unblemished macaroni with smooth outer surface.',
            'a photo of a macaroni showing uniform surface texture with no spots.',
            'a photo of a normal macaroni with intact hollow center.',
            'a photo of a macaroni with even color and no dark marks.',
            'a photo of a macaroni with proper shape and no deformation.',
            'a photo of a pristine macaroni with no contamination or foreign particles.',
            'a photo of a macaroni in perfect condition for food quality inspection.',
            'a photo of a good macaroni with smooth edges at both openings.',
            'a photo of a macaroni with clean surface and no visible defects.',
        ],
        'abnormal': [
            'a photo of a macaroni with a visible crack or fracture in the body.',
            'a photo of a macaroni with dark spots or discolored patches.',
            'a photo of a macaroni with broken ends or crumbled sections.',
            'a photo of a macaroni with deformed shape or crushed structure.',
            'a photo of a macaroni with surface contamination or foreign material.',
            'a photo of a macaroni with uneven color indicating quality issues.',
            'a photo of a macaroni with rough or pitted surface texture.',
            'a photo of a macaroni with holes or thin spots in the wall.',
            'a photo of a macaroni with irregular shape from production defects.',
            'a photo of a defective macaroni with visible manufacturing flaws.',
        ],
    },

    # ----------------------------------------------------------
    # PCB — defects: component defects, soldering issues, missing parts
    # ----------------------------------------------------------
    'pcb': {
        'normal': [
            'a photo of a defect-free PCB with properly soldered components.',
            'a photo of a PCB with all components correctly placed in position.',
            'a photo of a flawless PCB with clean solder joints and no bridges.',
            'a photo of a perfect PCB with intact copper traces and no breaks.',
            'a photo of a PCB without any missing components or empty pads.',
            'a photo of an unblemished PCB with uniform green solder mask.',
            'a photo of a PCB showing properly aligned components with no rotation.',
            'a photo of a normal PCB with clean board surface and no contamination.',
            'a photo of a PCB with consistent solder quality across all joints.',
            'a photo of a PCB with all capacitors and resistors in correct orientation.',
            'a photo of a pristine PCB with no burn marks or heat damage.',
            'a photo of a PCB in perfect condition for electronics quality inspection.',
            'a photo of a good PCB with intact silkscreen markings.',
            'a photo of a PCB with clean pads and properly formed solder connections.',
        ],
        'abnormal': [
            'a photo of a PCB with a missing component leaving an empty pad.',
            'a photo of a PCB with a solder bridge connecting adjacent pins.',
            'a photo of a PCB with a misaligned or rotated component.',
            'a photo of a PCB with cold solder joints showing poor wetting.',
            'a photo of a PCB with burned or heat-damaged area on the board.',
            'a photo of a PCB with scratched or broken copper traces.',
            'a photo of a PCB with excess solder forming blobs on the surface.',
            'a photo of a PCB with contamination or flux residue on the board.',
            'a photo of a PCB with a cracked or damaged component.',
            'a photo of a defective PCB with visible soldering or assembly flaws.',
        ],
    },

    # ----------------------------------------------------------
    # PIPE FRYUM — defects: surface anomalies
    # ----------------------------------------------------------
    'pipe fryum': {
        'normal': [
            'a photo of a defect-free pipe fryum with smooth tubular crispy surface.',
            'a photo of a pipe fryum with uniform golden color throughout.',
            'a photo of a flawless pipe fryum with proper hollow cylindrical shape.',
            'a photo of a perfect pipe fryum with consistent wall thickness.',
            'a photo of a pipe fryum without any cracks or broken sections.',
            'a photo of an unblemished pipe fryum with even crispy texture.',
            'a photo of a pipe fryum showing uniform expansion with no flat areas.',
            'a photo of a normal pipe fryum with consistent color and no burn spots.',
            'a photo of a pipe fryum with intact tubular structure.',
            'a photo of a pipe fryum with clean surface and no contamination.',
            'a photo of a pristine pipe fryum with even frying throughout.',
            'a photo of a pipe fryum in perfect condition for food quality inspection.',
            'a photo of a good pipe fryum with proper length and no breakage.',
            'a photo of a pipe fryum with natural appearance and no visible defects.',
        ],
        'abnormal': [
            'a photo of a pipe fryum with a visible crack along the tube.',
            'a photo of a pipe fryum with dark burn spots or charred areas.',
            'a photo of a pipe fryum with broken or snapped sections.',
            'a photo of a pipe fryum with collapsed structure that lost its shape.',
            'a photo of a pipe fryum with discolored patches from uneven frying.',
            'a photo of a pipe fryum with foreign material on the surface.',
            'a photo of a pipe fryum with holes in the wall of the tube.',
            'a photo of a pipe fryum with rough or blistered surface texture.',
            'a photo of a pipe fryum with irregular shape from production defects.',
            'a photo of a defective pipe fryum with visible quality issues.',
        ],
    },

    # ----------------------------------------------------------
    # CAPSULES (ViSA version) — similar to MVTec capsule
    # ----------------------------------------------------------
    'capsules': {
        'normal': [
            'a photo of a defect-free capsule with smooth transparent gelatin shell.',
            'a photo of a capsule with uniform fill level and consistent color.',
            'a photo of a flawless capsule with properly sealed seam.',
            'a photo of a perfect capsule with intact oblong shape.',
            'a photo of a capsule without any cracks or leaking contents.',
            'a photo of an unblemished capsule with clear surface and no haze.',
            'a photo of a capsule showing even gelatin thickness throughout.',
            'a photo of a normal capsule with properly aligned cap and body.',
            'a photo of a capsule with clean surface free of spots or marks.',
            'a photo of a capsule with consistent transparency and no bubbles.',
            'a photo of a pristine capsule with no deformation or squeeze marks.',
            'a photo of a capsule in perfect condition for pharmaceutical inspection.',
            'a photo of a good capsule with uniform color fill inside.',
            'a photo of a capsule with intact shell and proper seal.',
        ],
        'abnormal': [
            'a photo of a capsule with a crack or split in the gelatin shell.',
            'a photo of a capsule with leaking contents from a damaged seam.',
            'a photo of a capsule with discolored or cloudy patches on the shell.',
            'a photo of a capsule with a dented or squeezed shape.',
            'a photo of a capsule with bubbles or voids in the gelatin.',
            'a photo of a capsule with misaligned cap and body halves.',
            'a photo of a capsule with surface contamination or spots.',
            'a photo of a capsule with uneven fill level inside.',
            'a photo of a capsule with scratches or abrasions on the shell.',
            'a photo of a defective capsule with visible manufacturing defects.',
        ],
    },

    # ============================================================
    # GENERIC FALLBACK — for "object" and any unknown class
    # ============================================================
    'object': {
        'normal': [
            'a photo of a defect-free object with normal appearance.',
            'a photo of a flawless object with no visible damage.',
            'a photo of a perfect object without any defects.',
            'a photo of an unblemished object with intact surface.',
            'a photo of an object without flaw or damage.',
            'a photo of an object without any defect or anomaly.',
            'a photo of an object with clean and normal surface condition.',
            'a photo of a normal object in good condition.',
            'a photo of an object with no scratches or marks.',
            'a photo of an object with uniform appearance and no irregularities.',
            'a photo of a pristine object with no contamination.',
            'a photo of an object in perfect condition for visual inspection.',
            'a photo of a good object with no visible quality issues.',
            'a photo of an object with intact structure and no damage.',
        ],
        'abnormal': [
            'a photo of a damaged object with visible defects on the surface.',
            'a photo of a broken object with cracks or fractures.',
            'a photo of an object with a flaw or anomaly on the surface.',
            'a photo of an object with scratches or abrasion marks.',
            'a photo of an object with contamination or foreign material.',
            'a photo of an object with discoloration or staining.',
            'a photo of an object with a hole or missing section.',
            'a photo of an object with deformation or irregular shape.',
            'a photo of an object with surface damage indicating a defect.',
            'a photo of a defective object with visible quality problems.',
        ],
    },
}

# Map variant names to their base class prompts
_variant_mapping = {
    'macaroni1': 'macaroni',
    'macaroni2': 'macaroni',
    'pcb1': 'pcb',
    'pcb2': 'pcb',
    'pcb3': 'pcb',
    'pcb4': 'pcb',
}

def get_prompts(class_name):
    """
    Get normal and abnormal prompts for a given class name.
    Falls back to 'object' generic prompts for unknown classes.
    """
    # Check variant mapping first
    base_name = _variant_mapping.get(class_name, class_name)
    if base_name in class_specific_prompts:
        return class_specific_prompts[base_name]
    return class_specific_prompts['object']
