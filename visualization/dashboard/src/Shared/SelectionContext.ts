import uuidv4 from 'uuid/v4';

export interface ISelectionContextSubscriptions {
    selectionCallback: (s: string[]) => void;
    hoverCallback?: (s: string | undefined) => void;
    listenerCountCallback?: (n: number) => void;
}
export class SelectionContext {
    public selectedIds: string[] = [];
    public hoveredId: string | undefined;
    public listenerCount: number = 0;
    public readonly propertyOfInterest: string;

    private selectionSubs: Map<string, (s: string[]) => void> = new Map();
    private hoverSubs: Map<string, (s: string | undefined) => void> = new Map();
    private countSubs: Map<string, (n: number) => void> = new Map();
    private maxItems: number | undefined;

    constructor(propertyOfInterest: string, maxItems?: number) {
        this.propertyOfInterest = propertyOfInterest;
        this.maxItems = maxItems;
    }

    public onSelect(newSelections: string[]): void {
        // Keep the most recently added selections
        if (this.maxItems !== undefined) {
            newSelections = newSelections.slice(Math.max(newSelections.length - this.maxItems, 0));
        }
        this.selectedIds = newSelections;
        this.selectionSubs.forEach(sub => sub(newSelections));
    }

    public onHover(newItem: string | undefined): void {
        this.hoveredId = newItem;
        this.hoverSubs.forEach(sub => sub(newItem));
    }

    public subscribe(subs: ISelectionContextSubscriptions): string {
        const id = uuidv4();
        this.selectionSubs.set(id, subs.selectionCallback);

        if (subs.hoverCallback) {
            this.hoverSubs.set(id, subs.hoverCallback);
        }

        if (subs.listenerCountCallback) {
            this.countSubs.set(id, subs.listenerCountCallback);
        }

        this.listenerCount += 1;
        this.countSubs.forEach(val => val(this.listenerCount));
        return id;
    }

    public unsubscribe(id: string): void {
        this.selectionSubs.delete(id);
        this.hoverSubs.delete(id);
        this.countSubs.delete(id);
        this.listenerCount -= 1;
    }
}
